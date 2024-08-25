from influxdb_client import InfluxDBClient


class InfluxAPI:
    def __init__(self, url, token, org, bucket):
        # Influxdb configuration
        self.bucket = bucket
        self.org = org
        self.client = InfluxDBClient(
            url=url,
            token=token,
            org=org,
        )

    def save(self, point):
        try:
            write_api = self.client.write_api(write_options=SYNCHRONOUS)
            write_api.write(
                bucket=self.bucket,
                org=self.org,
                record=point,
            )
        except Exception as e:
            logging.error(f"Could not save data to Influx database {e}")
