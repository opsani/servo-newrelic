from __future__ import annotations
import datetime
import functools
import dateutil
import re
import collections
from typing import Awaitable, Callable, Dict, Iterable, List, Optional, Set, Tuple

import httpx
import pydantic
import servo


DEFAULT_BASE_URL = 'https://api.newrelic.com'
API_PATH = "/v2"

class NewrelicMetric(servo.Metric):
    """NewrelicMetric objects describe metrics that can be measure from the Newrelic APM API."""    
    
    query: pydantic.constr(
        regex=r"^[a-zA-Z]+:[a-z_]+$"
    )
    """The name and value of the APM data containing this Metric, seperated by a colon `name:value`

    For details, see the [TODO better Newrelic resource](https://docs.newrelic.com/docs/apis/rest-api-v2/application-examples-v2/average-response-time-examples-v2) documentation.
    """

    def __check__(self) -> servo.Check:
        return servo.Check(
            name=f"Check {self.name}",
            description=f"Run Newrelic get \"{self.fetch_name}: {self.values_selector}\""
        ) # TODO Checker class

class NewrelicConfiguration(servo.BaseConfiguration):
    """NewrelicConfiguration objects describe how NewrelicConnector objects
capture measurements from the Newrelic metrics server. 
    """

    # TODO: only needed for insights API query support
    # account_id: str
    # """The Account ID for accessing the Newrelic metrics API."""

    app_id: str
    """The Application ID for accessing the Newrelic metrics API."""

    api_key: pydantic.SecretStr
    """The API key for accessing the Newrelic metrics API."""

    base_url: pydantic.AnyHttpUrl = DEFAULT_BASE_URL
    """The base URL for accessing the Newrelic metrics API.

    The URL must point to the root of the Newrelic deployment. Resource paths
    are computed as necessary for API requests.
    """

    metrics: List[NewrelicMetric]
    """The metrics to measure from Newrelic.

    Metrics must include a valid fetch object and values selector.
    """

    step: servo.Duration = "1m"
    """The resolution of the metrics to be fetched.
    
    The step resolution determines the number of data points captured across a
    query range.
    """

    hostname_filter_format: str = "ip-{}.us-west-2.compute.internal"
    """Used when responding to the instances event, determines how the `filter[hostname]` string is formatted
    """

    @classmethod # TODO
    def generate(cls, **kwargs) -> "NewrelicConfiguration":
        """Generates a default configuration for capturing measurements from the
        Newrelic metrics server.

        Returns:
            A default configuration for NewrelicConnector objects.
        """
        return cls(
            description="Update the base_url and metrics to match your Newrelic configuration",
            metrics=[
                NewrelicMetric(
                    name="throughput",
                    unit=servo.Unit.REQUESTS_PER_SECOND,
                    fetch_name="HttpDispatcher",
                    values_selector="requests_per_minute"
                ),
                NewrelicMetric(
                    "error_rate", servo.Unit.COUNT, "Errors/all", "error_count"
                ),
            ],
            **kwargs,
        )
    
    @pydantic.validator("base_url")
    @classmethod
    def rstrip_base_url(cls, base_url):
        return base_url.rstrip("/")

    @property
    def api_url(self) -> str:
        return f"{self.base_url}{API_PATH}"

class NewrelicRequest(pydantic.BaseModel):
    base_url: pydantic.AnyHttpUrl
    fetches: Dict[str, Set[str]]
    start: datetime
    end: datetime
    step: servo.Duration

    @property
    def params(self) -> Dict[str, str]:
        return { 
            'names[]': '+'.join(self.fetches.keys()),
            'values[]': '+'.join(functools.reduce(lambda x, y: x+y, self.fetches.values())),
            'from': self.start.isoformat(),
            'to': self.end.isoformat(),
            'period': self.step.total_seconds(),
            'summarize': False,
            'raw': True,
        }

class NewrelicChecks(servo.BaseChecks):
    """NewrelicChecks objects check the state of a NewrelicConfiguration to
determine if it is ready for use in an optimization run.
    """
    config: NewrelicConfiguration

    @servo.require("Connect to \"{self.config.base_url}\"")
    async def check_base_url(self) -> None:
        """Checks that the Newrelic base URL is valid and reachable.
        """
        async with httpx.AsyncClient(base_url=self.config.api_url) as client:
            response = await client.get(f"applications/{self.config.app_id}.json") # TODO may not always have permissions to do this
            response.raise_for_status()

    # TODO: how to structure multicheck to iterate a list of instances and request multiple NR metrics in a single request
    # @servo.multicheck("Run fetch \"{item.query}\"")
    # async def check_fetches(self) -> Tuple[Iterable, CheckHandler]:
    #     """Checks that all metrics have valid, well-formed PromQL queries.
    #     """
    #     async def query_for_metric(metric: NewrelicMetric) -> str:
    #         start, end = datetime.now() - timedelta(minutes=10), datetime.now()
    #         newrelic_request = NewrelicRequest(base_url=self.config.api_url, metric=metric, start=start, end=end)

    #         logger.trace(f"Querying Newrelic (`{metric.query}`): {newrelic_request.url}")
    #         async with httpx.AsyncClient() as client:
    #             response = await client.get(newrelic_request.url)
    #             response.raise_for_status()
    #             result = response.json()
    #             return f"returned {len(result)} results"

    #     return self.config.metrics, query_for_metric

@servo.metadata(
    description="Newrelic Connector for Opsani",
    version="0.0.1",
    homepage="https://github.com/opsani/newrelic-connector",
    license=servo.License.APACHE2,
    maturity=servo.Maturity.EXPERIMENTAL,
)
class NewrelicConnector(servo.BaseConnector):
    """NewrelicConnector objects enable servo assemblies to capture
    measurements from the [Newrelic](https://newrelic.com/) metrics server.
    """
    config: NewrelicConfiguration

    # TODO
    # @on_event()
    # async def check(self,
    #     filter_: Optional[Filter] = None, 
    #     halt_on: HaltOnFailed = HaltOnFailed.requirement
    # ) -> List[Check]:
    #     """Checks that the configuration is valid and the connector can capture        
    #     measurements from Newrelic.

    #     Checks are implemented in the NewrelicChecks class.

    #     Args:
    #         filter_ (Optional[Filter], optional): A filter for limiting the
    #             checks that are run. Defaults to None.
    #         halt_on (HaltOnFailed, optional): When to halt running checks.
    #             Defaults to HaltOnFailed.requirement.

    #     Returns:
    #         List[Check]: A list of check objects that report the outcomes of the            
    #             checks that were run.
    #     """        
    #     return await NewrelicChecks.run(self.config, filter_, halt_on=halt_on)

    @servo.on_event()
    def describe(self) -> servo.Description:
        """Describes the current state of Metrics measured by querying Newrelic.

        Returns:
            Description: An object describing the current state of metrics
                queried from Newrelic.
        """
        return servo.Description(metrics=self.config.metrics)

    @property
    @servo.on_event()
    def metrics(self) -> List[servo.Metric]:
        """Returns the list of Metrics measured through Newrelic queries.

        Returns:
            List[Metric]: The list of metrics to be queried.
        """
        return self.config.metrics

    @servo.on_event()
    async def instances(self) -> List[str]:
        async with self.api_client as client:
            try:
                response = await client.get('/config')
                # response = await client.get(f'/assets/opsani.com/{component}') # TODO
                response.raise_for_status()
            except (httpx.HTTPError, httpx.ReadTimeout, httpx.ConnectError) as error:                
                self.logger.trace(f"HTTP error encountered during oco config GET {self.optimizer.api_url}/config: {error}")
                raise

        opsani_config = response.json()
        canary = opsani_config.get('adjustment', {}).get('control', {}).get('userdata', {}).get('canary')
        # canary = opsani_config.get('data') # TODO
        canary_ip = canary.get('ip')
        canary_ip = re.sub(r"\.", "-", canary_ip)
        canary_hostname = self.config.hostname_filter_format.format(canary_ip)

        async with httpx.AsyncClient(
            base_url=self.config.api_url + 'applications/{newrelic_app_id}/instances.json'.format(self.config.app_id), 
            params={'filter[hostname]': canary_hostname},
            headers={'X-Api-Key': self.config.api_key},
        ) as client:
            try:
                response = await client.get()
                response.raise_for_status()
            except (httpx.HTTPError, httpx.ReadTimeout, httpx.ConnectError) as error:                
                self.logger.trace(f"HTTP error encountered during GET {error.request.url}: {error}")
                raise

        return [ai['host'] for ai in response.json()['application_instances']] 


    @servo.on_event()
    async def measure(
        self, *, metrics: List[str] = None, control: servo.Control = servo.Control()
    ) -> servo.Measurement:
        """Queries Newrelic for metrics as time series values and returns a
        Measurement object that aggregates the readings for processing by the
        optimizer.

        Args:
            metrics (List[str], optional): A list of the metric names to measure. 
                When None, all configured metrics are measured. Defaults to None.
            control (Control, optional): A control descriptor that describes how            
                the measurement is to be captured. Defaults to Control().

        Returns:
            Measurement: An object that aggregates the state of the metrics
            queried from Newrelic.
        """
        if metrics:
            metrics__ = list(filter(lambda m: m.name in metrics, self.metrics))
        else:
            metrics__ = self.metrics
        measuring_names = list(map(lambda m: m.name, metrics__))
        self.logger.info(f"Starting measurement of {len(metrics__)} metrics: {servo.utilities.join_to_series(measuring_names)}")

        start = datetime.now() + control.warmup
        end = start + control.duration

        sleep_duration = servo.Duration(control.warmup + control.duration)
        self.logger.info(
            f"Waiting {sleep_duration} during metrics collection ({control.warmup} warmup + {control.duration} duration)..."
        )

        progress = servo.DurationProgress(sleep_duration)
        notifier = lambda p: self.logger.info(p.annotate(f"waiting {sleep_duration} during metrics collection...", False), progress=p.progress)
        await progress.watch(notifier)
        self.logger.info(f"Done waiting {sleep_duration} for metrics collection, resuming optimization.")

        # Fetch the measurements
        self.logger.info(f"Querying Newrelic for {len(metrics__)} metrics...")
        readings = await self._query_newrelic(metrics__, start, end)
        measurement = servo.Measurement(readings=readings)
        return measurement

    async def _query_newrelic(
        self, metrics: List[NewrelicMetric], start: datetime, end: datetime
    ) -> List[servo.TimeSeries]:
        f_names_to_ms: Dict[str, List[NewrelicMetric]] = collections.defaultdict(list)
        fetches: Dict[str, Set[str]] = servo.defaultdict(set)
        for m in metrics:
            fetch_name, values_selector = m.query.split(':')
            f_names_to_ms[fetch_name].append(m)
            fetches[fetch_name].add(values_selector)

        newrelic_request = NewrelicRequest(base_url=self.config.api_url, fetches=fetches, start=start, end=end, step=self.config.step)

        self.logger.trace(f"Getting Newrelic instance ids: {newrelic_request.url}")
        instance_ids: List[str] = await self.dispatch_event("instances")
        
        self.logger.trace(f"Querying Newrelic: {newrelic_request.url}")
        readings = []
        # TODO asyncio gather this instead
        for i in instance_ids:
            api_path = '/applications/{app_id}/instances/{instance_id}/metrics/data.json'.format(self.config.app_id, i)
            self.logger.trace(f"Querying Newrelic for instance: {i}")
            async with httpx.AsyncClient(
                base_url=self.config.api_url + api_path, 
                params=newrelic_request.params,
                headers={'X-Api-Key': self.config.api_key},
            ) as client:
                try:
                    response = await client.get()
                    response.raise_for_status()
                except (httpx.HTTPError, httpx.ReadTimeout, httpx.ConnectError) as error:                
                    self.logger.trace(f"HTTP error encountered during GET {newrelic_request.url}: {error}")
                    raise

            data = response.json()
            self.logger.trace(f"Got response data for instance {i}: {data}")

            for fetched_m in data.get('metric_data', {}).get('metrics', []):
                m_readings: Dict[NewrelicMetric, List[Tuple[datetime.datetime, servo.Numeric]]] = collections.defaultdict(list)
                for m in f_names_to_ms[fetched_m['name']]:
                    _, values_selector = m.query.split(':')
                    for ts in fetched_m.get('timeslices', []):
                        m_readings[m].append((dateutil.parser.parse(ts['from']), ts['values'][values_selector]))

                        readings.append(
                            servo.TimeSeries(
                                metric=m,
                                values=m_readings[m],
                                id=i,
                                metadata=dict(instance=i)
                            )
                        )
        
        return readings
