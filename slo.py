#!/usr/bin/env python3

import abc
from collections import defaultdict
from copy import copy
from datetime import datetime, timedelta, timezone
import decimal
import operator
import pydantic
import statistics
from typing import Any, Callable, Dict, List, Optional, Union
import time

class FastFailConfig(pydantic.BaseModel):
    disabled: pydantic.conint(ge=0, le=1, multiple_of=1) = 0
    period: timedelta = timedelta(seconds=60)
    span: timedelta = None
    skip: timedelta = timedelta(seconds=0)
    
    class Config:
        extra = 'forbid'

    @pydantic.validator('span', pre=True, always=True)
    def span_defaults_to_period(cls, v, *, values, **kwargs):
        if v is None:
            return values['period']
        return v


class BaseSloCondition(pydantic.BaseModel, abc.ABC):
    description: Optional[str] = None
    metric: str
    threshold_multiplier: decimal.Decimal = decimal.Decimal(1)
    keep: pydantic.constr(strip_whitespace=True, regex='above|below') = 'below'
    for_: int = pydantic.Field(default=1, alias='for', ge=1)
    
    class Config:
        extra = 'forbid'

    @abc.abstractclassmethod
    def get_values(self, metrics: Dict[str, Any], result: "SloResult") -> bool:
        """Produce all scalar values needed to perform the configured check or return False to indicate
        further processing is not possible

        Args:
            metrics (Dict[str, Any]): The retrieved metrics to evaluate this condition against
            result (SloResult): Object onto which condition information is marshalled for computation and reporting.
                Most importantly, `result.metric_scalar` and `result.threshold_product` will always be set on a successful
                (True) result. Other properties set are purely for reporting and debug purposes

        Returns:
            bool: True if the values are ready to check, False to indicate further processing is not possible
        """
        ...

    def __str__(self) -> str:
        if self.description is None:
            return f"({self.metric})"
        return f"({self.metric} -> {self.description})"

    def check_values(self, metric_value: decimal.Decimal, threshold_value: decimal.Decimal) -> bool:
        """Compares the provided metric and threshold decimal values based on the configured keep

        Returns: bool: Indicates whether the values satisfy the configured keep condition
        """
        check_op = operator.lt if self.keep == 'below' else operator.gt
        return check_op(metric_value, threshold_value)

    def _get_metric_value(self, metrics, result: "SloResult") -> bool:
        """Produce the scalar value for the configured metric and store into result.metric_scalar. 
        Should only be invoked by a subclass in get_values

        Returns: bool: True if metric value is ready to check, False to indicate further processing is not possible
        """
        if self.metric not in metrics:
            result.message = 'Condition metric not contained in fetched metrics'
            return False

        metric_values = metrics[self.metric]['values']
        if not metric_values:
            result.message = 'Condition metric values contained empty list of instances'
            return False

        result.instances = ', '.join([val['id'] for val in metric_values])

        instance_values = [val['data'] for val in metric_values]
        # Check at least one list has values
        if not any(instance_values):
            result.message = 'Condition metric instance values all had empty data'
            return False

        inst_scalars = []
        for val_list in instance_values:
            if (scalar := get_scalar_from_timeseries(val_list)) is not None:
                inst_scalars.append(scalar)

        if not inst_scalars:
            result.message = 'Condition metric instance values all had 0/falsey data'
            return False

        result.metric_scalar = inst_scalars[0] if len(inst_scalars) == 1 else statistics.mean(inst_scalars)
        return True

def get_scalar_from_timeseries(timeseries: List[List[float]]) -> Optional[decimal.Decimal]:
    # Check for empty
    if not timeseries:
        return None

    # Pull metric values from timestamps, filter out zero values to exclude timeslices for which no data was reported
    ts_vals = [decimal.Decimal(v[1]) for v in timeseries if v[1]]
    if not ts_vals:
        return None

    if len(ts_vals) == 1:
        return ts_vals[0]
    else:
        # Compute mean for current instance
        return statistics.mean(ts_vals)

class ThresholdConstantCondition(BaseSloCondition):
    threshold: decimal.Decimal

    def get_values(self, metrics, result: "SloResult") -> bool:
        if not self._get_metric_value(metrics, result):
            return False

        result.threshold_product = self.threshold * self.threshold_multiplier
        return True

class ThresholdMetricCondition(BaseSloCondition):
    threshold_metric: str

    def get_values(self, metrics, result: "SloResult") -> bool:
        if not self._get_metric_value(metrics, result):
            return False

        if self.threshold_metric not in metrics:
            result.message = 'Condition threshold metric not contained in fetched metrics'
            return False

        metric_values = metrics[self.threshold_metric]['values']
        if not metric_values:
            result.message = 'Condition threshold metric values contained empty list of instances'
            return False

        result.threshold_instances = ', '.join([val['id'] for val in metric_values])

        instance_values = [val['data'] for val in metric_values]
        # Check at least one list has values
        if not any(instance_values):
            result.message = 'Condition threshold metric instance values all had empty data'
            return False

        inst_scalars = []
        for val_list in instance_values:
            if (scalar := get_scalar_from_timeseries(val_list)) is not None:
                inst_scalars.append(scalar)

        if not inst_scalars:
            result.message = 'Condition threshold metric instance values all had 0/falsey data'
            return False

        result.threshold_metric_scalar = inst_scalars[0] if len(inst_scalars) == 1 else statistics.mean(inst_scalars)
        result.threshold_product = result.threshold_metric_scalar * self.threshold_multiplier
        return True

class SloInput(pydantic.BaseModel):
    conditions: List[Union[ThresholdConstantCondition, ThresholdMetricCondition]]

# Inherit from pydantic model for simple json serialization
class SloResult(pydantic.BaseModel):
    index: int
    condition: Union[ThresholdConstantCondition, ThresholdMetricCondition]
    message: str = ''
    metric_scalar: Optional[decimal.Decimal] = None
    threshold_metric_scalar: Optional[decimal.Decimal] = None
    threshold_product: Optional[decimal.Decimal] = None
    instances: Optional[str] = None
    threshold_instances: Optional[str] = None
    failed_count: int = 0 # Only updated on calls to check_counter

    def __str__(self):
        x_num = ''
        if self.failed_count:
            x_num = f' x{self.failed_count}'
        return f"{self.condition} {self.message}{x_num}. computed values: metric {self.metric_scalar} threshold {self.threshold_product}"

    def check_for(self, failed_count: int):
        self.failed_count = failed_count
        return failed_count >= self.condition.for_


class SloResults(pydantic.BaseModel):
    """Represents the results of a single check run

    Attributes:
        from_ (datetime): The start of the time span for which the check metrics were queried
        to (datetime): The end of the time span for which the check metrics were queried
        passed (List[SloResult]): List of SLO Check results which passed for the provided metrics
        failed (List[SloResult]): List of SLO Check results which failed for the provided metrics
        missing (List[SloResult]): List of SLO Check results which could not be computed for the provided metrics
        slo_input (SloInput): Configuration of SLO check conditions provided in the control section of driver input
        metrics (Dict[str, Any]): Queried metrics used to determine whether checks are passing
    """

    from_: datetime
    to: datetime
    passed: List[SloResult] = []
    failed: List[SloResult] = []
    missing: List[SloResult] = []

    @property
    def failed_message(self) -> str:
        """Returns a string suitable for reporting the current set of failures
        """
        return f"[from {self.from_} to {self.to}]" + ' | '.join(map(str, self.failed))

    def __init__(self, slo_input: SloInput, metrics: Dict[str, Any], *args, **kwargs):
        super().__init__(*args, **kwargs)

        for index, condition in enumerate(slo_input.conditions):
            # Index is used as a psuedo ID to track failures across check_sleep calls
            result = SloResult(condition=condition, index=index)

            # Aggregate metrics, prepare result for calculation
            if not condition.get_values(metrics, result):
                self.missing.append(result)
                continue

            # Perform checks
            if condition.check_values(result.metric_scalar, result.threshold_product):
                result.message = 'SLO passed'
                self.passed.append(result)
            else:
                result.message = f'keep {condition.keep} not met'
                self.failed.append(result)

    def check_counters(self, counts_dict: Dict[int, int]) -> bool:
        """Updates an external dictionary of counters based on the current set of failures.
        Stores updated counts onto applicable failed results
        """
        new_counts, should_raise = {}, False
        for fr in self.failed:
            new_counts[fr.index] = counts_dict[fr.index] + 1
            if fr.check_for(new_counts[fr.index]):
                should_raise = True

        # Remove counters from previous failures that didn't fail this cycle
        counts_dict.clear()
        counts_dict.update(new_counts)

        return should_raise


class SloChecker(pydantic.BaseModel):
    """This class serves as the interface between drivers and most SLO Check logic. It is initialized with the
    necessary components to determine the state of configured checks and provides simple interfaces for detecting and
    reporting failures

    Attributes:
        timezone (timezone): Allows overriding of the time zone for the date parameters of the metrics query
        _failed_counters (Dict[int, int]): Maps index of slo_input condition to the number of times it has failed during
            sleep_check. Updated by reference inside SloResults.check_counters
    """
    fast_fail_config: FastFailConfig
    slo_input: SloInput
    timezone_: Optional[timezone] = pydantic.Field(None, alias='timezone')

    metrics_getter: Callable[[datetime,datetime], Dict[str, Any]]
    on_pass: Callable[["SloChecker"], None]
    on_fail: Callable[["SloChecker"], None]

    _skip_remaining: float = pydantic.PrivateAttr()
    _failed_counters: Dict[int, int] = pydantic.PrivateAttr(default_factory=lambda: defaultdict(int))

    # TODO: track list of results for which there are active counters
    last_results: Optional[SloResults] = None

    def json(self):
        return super().json(exclude={'slo_input', 'metrics_getter', 'on_pass', 'on_fail'})

    class Config:
        arbitrary_types_allowed = True

        json_encoders = {
            timezone: str,
        }

    @property
    def failed_message(self):
        if not self.last_results:
            return 'failed results not set'
        return self.last_results.failed_message

    def __init__(self, slo_input, *args, **kwargs):
        if isinstance(slo_input, dict):
            slo_input = SloInput.parse_obj(slo_input)
        super().__init__(*args, **kwargs, slo_input=slo_input)
        self._skip_remaining = self.fast_fail_config.skip.total_seconds()

    def error_json(self, status='failed'):
        return SloError(
            message=self.failed_message, 
            reason='slo-violation', 
            status=status,
            checks=self
        ).json()

    def check(self, single_check=True) -> SloResults:
        measure_to = datetime.now(self.timezone_).replace(microsecond=0)
        measure_from = (measure_to - self.fast_fail_config.span).replace(microsecond=0)
        
        metrics = self.metrics_getter(measure_from, measure_to)
        results = SloResults(
            from_=measure_from, 
            to=measure_to, 
            slo_input=self.slo_input, 
            metrics=metrics
        )

        if single_check:
            self.last_results = results
            self.last_results.check_counters(self._failed_counters)
            if results.failed:
                self.on_fail(self)
            else:
                self.on_pass(self)
        
        return results

    def check_sleep(self, secs: float) -> None:
        """Pass through to time.sleep that divides the duration so that SLO checks are run
        periodically based on the configuration of the fast_fail period
        """
        end_at = datetime.now() + timedelta(seconds=secs)
        # Skip SLO checks for configured duration
        if self._skip_remaining:
            if self._skip_remaining > secs:
                self._skip_remaining -= secs
                time.sleep(secs)
                return
            time.sleep(self._skip_remaining)
            self._skip_remaining = 0

        while (loop_start := datetime.now()) < end_at:
            self.last_results = self.check(single_check=False)
            if self.last_results.check_counters(self._failed_counters):
                self.on_fail(self)
            else:
                self.on_pass(self)

            # account for time taken by SLO check, clamp negative timedeltas to 0
            remaining_period = max(0, (self.fast_fail_config.period - (datetime.now() - loop_start)).total_seconds())
            remaining_total = max(0, (end_at - datetime.now()).total_seconds())
            next_sleep = min(remaining_period, remaining_total)
            time.sleep(next_sleep)

# Define model for error to leverage pydantic json encoding on sub-properties
class SloError(pydantic.BaseModel):
    status: str = 'failed'
    reason: str
    message: str
    checks: Optional[SloChecker] = None

    def json(self):
        return super().json(exclude={'checks': {'slo_input', 'metrics_getter', 'on_pass', 'on_fail'}})

    class Config:
        json_encoders = { timezone: str }
