# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

import logging
from typing import ClassVar, Optional, Sequence, Callable

import pandas as pd
from pyhive import hive
from pyhive import exc as pyhive_exc
from sqlalchemy.engine import make_url

from metricflow.dataflow.sql_table import SqlTable
from metricflow.logging.formatting import indent_log_line
from metricflow.object_utils import pformat_big_objects
from metricflow.protocols.async_sql_client import AsyncSqlClient
from metricflow.protocols.sql_client import SqlEngine, SqlEngineAttributes, SqlIsolationLevel
from metricflow.protocols.sql_client import SqlClient
from metricflow.protocols.sql_request import SqlRequestId
from metricflow.protocols.sql_request import SqlRequestTagSet, SqlJsonTag
from metricflow.sql.sql_bind_parameters import SqlBindParameters
from metricflow.sql_clients.async_request import SqlStatementCommentMetadata, CombinedSqlTags
from metricflow.sql_clients.base_sql_client_implementation import BaseSqlClientImplementation
from metricflow.sql_clients.common_client import SqlDialect, check_isolation_level, not_empty
from metricflow.sql_clients.sql_utils import SqlClientException
from metricflow.sql.render.spark_renderer import SparkSqlQueryPlanRenderer

logger = logging.getLogger(__name__)


class SparkThriftEngineAttributes:
    """Engine-specific attributes for the Spark Thrift query engine."""

    sql_engine_type: ClassVar[SqlEngine] = SqlEngine.SPARK
    sql_query_plan_renderer: ClassVar[SparkSqlQueryPlanRenderer] = SparkSqlQueryPlanRenderer()

    # SQL Engine capabilities
    supported_isolation_levels: ClassVar[Sequence[SqlIsolationLevel]] = ()
    date_trunc_supported: ClassVar[bool] = True  # Spark supports DATE_TRUNC
    full_outer_joins_supported: ClassVar[bool] = True  # Spark supports FULL OUTER JOIN
    indexes_supported: ClassVar[bool] = False  # Spark doesn't support traditional indexes
    multi_threading_supported: ClassVar[bool] = True
    timestamp_type_supported: ClassVar[bool] = True
    timestamp_to_string_comparison_supported: ClassVar[bool] = True
    cancel_submitted_queries_supported: ClassVar[bool] = False  # Limited support in Spark
    continuous_percentile_aggregation_supported: ClassVar[bool] = True
    discrete_percentile_aggregation_supported: ClassVar[bool] = True
    approximate_continuous_percentile_aggregation_supported: ClassVar[bool] = False
    approximate_discrete_percentile_aggregation_supported: ClassVar[bool] = False

    # SQL Dialect replacement strings
    double_data_type_name: ClassVar[str] = "DOUBLE"
    timestamp_type_name: ClassVar[Optional[str]] = "TIMESTAMP"
    random_function_name: ClassVar[str] = "RAND"


class SparkThriftSqlClient(BaseSqlClientImplementation):
    """SQL Client for Spark Thrift Server using PyHive."""

    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        database: str,
        auth: str = "LDAP",
        **kwargs,
    ) -> None:
        """Initialize Spark Thrift client.

        Args:
            host: Spark Thrift Server host
            port: Spark Thrift Server port (default: 10000)
            username: Username for authentication
            password: Password for authentication
            database: Default database name
            auth: Authentication method (NONE, LDAP, CUSTOM)
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.auth = auth
        self._connection: Optional[hive.Connection] = None

        super().__init__()

    @property
    def sql_engine_attributes(self) -> SqlEngineAttributes:
        """Returns Spark Thrift engine attributes."""
        return SparkThriftEngineAttributes

    @classmethod
    def from_connection_details(cls, url: str, password: Optional[str] = None) -> "SparkThriftSqlClient":  # noqa: D
        """Construct SparkThriftSqlClient from connection URL.

        URL format: spark_thrift://username@host:port/database
        Password should be provided via the password parameter.
        """
        parsed_url = make_url(url)
        dialect = SqlDialect.SPARK_THRIFT.value

        if parsed_url.drivername != dialect:
            raise ValueError(f"Expected dialect '{dialect}' in {url}")

        if password is None:
            raise ValueError(f"Password not supplied for {url}")

        return cls(
            host=not_empty(parsed_url.host, "host", url),
            port=not_empty(parsed_url.port, "port", url),
            username=not_empty(parsed_url.username, "username", url),
            password=password,
            database=not_empty(parsed_url.database, "database", url),
        )

    def _format_run_query_log_message(self, statement: str, sql_bind_parameters: SqlBindParameters) -> str:
        """Format query log message."""
        message = f"Running query:\n\n{indent_log_line(statement)}"
        if len(sql_bind_parameters.param_dict) > 0:
            message += (
                f"\n"
                f"\n"
                f"with parameters:\n"
                f"\n"
                f"{indent_log_line(pformat_big_objects(sql_bind_parameters.param_dict))}"
            )
        return message

    def _get_connection(self) -> hive.Connection:
        """Get or create PyHive connection."""
        if self._connection is None:
            try:
                logger.info(f"Creating PyHive connection to {self.host}:{self.port}")
                self._connection = hive.Connection(
                    host=self.host,
                    port=self.port,
                    username=self.username,
                    password=self.password,
                    database=self.database,
                    auth=self.auth,
                )
                logger.info("PyHive connection established successfully")
            except pyhive_exc.Error as e:
                logger.error(f"Failed to create PyHive connection: {e}")
                raise SqlClientException(f"Failed to connect to Spark Thrift Server: {e}") from e
        return self._connection

    def _engine_specific_query_implementation(
        self,
        stmt: str,
        bind_params: SqlBindParameters,
        isolation_level: Optional[SqlIsolationLevel] = None,
        system_tags: SqlRequestTagSet = SqlRequestTagSet(),
        extra_tags: SqlJsonTag = SqlJsonTag(),
    ) -> pd.DataFrame:
        """Execute query and return results as DataFrame."""
        check_isolation_level(self, isolation_level)

        connection = self._get_connection()
        cursor = connection.cursor()

        try:
            # Log the query
            logger.info(self._format_run_query_log_message(stmt, bind_params))

            # Execute query
            cursor.execute(stmt)

            # Fetch all results
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            # Convert to DataFrame
            df = pd.DataFrame(rows, columns=columns)

            logger.info(f"Query completed successfully, returned {len(df)} rows")
            return df

        except pyhive_exc.Error as e:
            logger.error(f"Query execution failed: {e}")
            raise SqlClientException(f"Query execution failed: {e}") from e

        finally:
            cursor.close()

    def _engine_specific_execute_implementation(
        self,
        stmt: str,
        bind_params: SqlBindParameters,
        isolation_level: Optional[SqlIsolationLevel] = None,
        system_tags: SqlRequestTagSet = SqlRequestTagSet(),
        extra_tags: SqlJsonTag = SqlJsonTag(),
    ) -> None:
        """Execute DDL/DML statement."""
        check_isolation_level(self, isolation_level)

        connection = self._get_connection()
        cursor = connection.cursor()

        try:
            # Log the statement
            logger.info(self._format_run_query_log_message(stmt, bind_params))

            # Execute statement
            cursor.execute(stmt)

            # Commit if necessary (PyHive auto-commits by default for DDL/DML)
            logger.info("Statement executed successfully")

        except pyhive_exc.Error as e:
            logger.error(f"Statement execution failed: {e}")
            raise SqlClientException(f"Statement execution failed: {e}") from e

        finally:
            cursor.close()

    def _engine_specific_dry_run_implementation(self, stmt: str, bind_params: SqlBindParameters) -> None:
        """Execute dry run - validate query without fetching results."""
        connection = self._get_connection()
        cursor = connection.cursor()

        try:
            # For dry run, we execute with LIMIT 0 or use EXPLAIN
            explain_stmt = f"EXPLAIN {stmt}"

            logger.info(f"Running dry run with: {explain_stmt}")

            cursor.execute(explain_stmt)

            # Fetch and discard results
            cursor.fetchall()

            logger.info("Dry run completed successfully")

        except pyhive_exc.Error as e:
            logger.error(f"Dry run failed: {e}")
            raise SqlClientException(f"Dry run failed: {e}") from e

        finally:
            cursor.close()

    def create_table_from_dataframe(
        self,
        sql_table: SqlTable,
        df: pd.DataFrame,
        chunk_size: Optional[int] = None,
    ) -> None:
        """Create table and insert data from DataFrame.

        Args:
            sql_table: Table to create
            df: DataFrame to insert
            chunk_size: Number of rows to insert per batch
        """
        if df.empty:
            logger.warning("Empty DataFrame, skipping table creation")
            return

        connection = self._get_connection()
        cursor = connection.cursor()

        try:
            # Drop table if exists
            cursor.execute(f"DROP TABLE IF EXISTS {sql_table.sql}")

            # Create table from DataFrame schema
            columns = []
            for col_name, dtype in df.dtypes.items():
                if pd.api.types.is_integer_dtype(dtype):
                    spark_type = "INT"
                elif pd.api.types.is_float_dtype(dtype):
                    spark_type = "DOUBLE"
                elif pd.api.types.is_datetime64_dtype(dtype):
                    spark_type = "TIMESTAMP"
                else:
                    spark_type = "STRING"
                columns.append(f"`{col_name}` {spark_type}")

            create_stmt = f"CREATE TABLE {sql_table.sql} ({', '.join(columns)})"
            cursor.execute(create_stmt)

            # Insert data in chunks
            chunk_size = chunk_size or 1000
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i + chunk_size]
                values = []
                for _, row in chunk.iterrows():
                    row_values = []
                    for val in row.values:
                        if pd.isna(val):
                            row_values.append("NULL")
                        elif isinstance(val, str):
                            escaped_val = val.replace("'", "''")
                            row_values.append(f"'{escaped_val}'")
                        else:
                            row_values.append(str(val))
                    values.append(f"({', '.join(row_values)})")

                insert_stmt = f"INSERT INTO {sql_table.sql} VALUES {', '.join(values)}"
                cursor.execute(insert_stmt)

            logger.info(f"Created table {sql_table.sql} with {len(df)} rows")

        except pyhive_exc.Error as e:
            logger.error(f"Failed to create table from DataFrame: {e}")
            raise SqlClientException(f"Failed to create table from DataFrame: {e}") from e

        finally:
            cursor.close()

    def close(self) -> None:
        """Close the PyHive connection."""
        if self._connection is not None:
            try:
                self._connection.close()
                logger.info("PyHive connection closed")
            except pyhive_exc.Error as e:
                logger.warning(f"Error closing connection: {e}")
            finally:
                self._connection = None

    def list_tables(self, schema_name: str) -> Sequence[str]:
        """List all tables in the given schema."""
        connection = self._get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute(f"SHOW TABLES IN {schema_name}")
            tables = [row[0] for row in cursor.fetchall()]
            return tables
        except pyhive_exc.Error as e:
            logger.error(f"Failed to list tables: {e}")
            raise SqlClientException(f"Failed to list tables: {e}") from e
        finally:
            cursor.close()

    def list_schemas(self) -> Sequence[str]:
        """List all schemas (databases)."""
        connection = self._get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute("SHOW DATABASES")
            schemas = [row[0] for row in cursor.fetchall()]
            return schemas
        except pyhive_exc.Error as e:
            logger.error(f"Failed to list schemas: {e}")
            raise SqlClientException(f"Failed to list schemas: {e}") from e
        finally:
            cursor.close()

    def cancel_request(self, match_function: Callable[[CombinedSqlTags], bool]) -> int:
        """Cancel requests that match the given tag function.

        Note: Spark Thrift Server has limited query cancellation support.
        This implementation returns 0 as cancellation is not reliably supported.

        Returns:
            Number of cancellation commands sent (always 0 for now)
        """
        logger.warning("Query cancellation is not reliably supported in Spark Thrift Server")
        # For now, return 0 as we cannot reliably cancel queries in Spark Thrift
        # Future: Could implement using session management if needed
        return 0

    def cancel_submitted_queries(self) -> None:
        """Cancel all submitted queries that have not yet started executing.

        Note: Spark Thrift Server has very limited query cancellation capabilities.
        This method is a no-op as we cannot reliably cancel submitted queries.
        """
        logger.warning("Query cancellation of submitted queries is not supported in Spark Thrift Server")
        # No-op: Spark Thrift doesn't provide API to cancel submitted queries
        return

    def active_requests(self) -> Sequence[SqlRequestId]:
        """Return requests that are still in progress.

        Note: Spark Thrift Server doesn't provide a way to track active queries.
        Returns empty list as we cannot reliably track active requests.
        """
        # Spark Thrift doesn't provide query tracking like some other engines
        # Return empty list as we cannot determine which queries are still active
        return []
