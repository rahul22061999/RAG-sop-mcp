"""
Tests for Settings.vector_store's connection-string construction.

Covers two regressions fixed during development:
- sslmode being hardcoded to "require" (broke against the local Docker
  Postgres container, which has no TLS cert configured).
- PGVectorStore.from_params() building its connection string with a raw
  f-string, silently mangling special characters (e.g. "@") in the password.
"""

from config.settings import Settings


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        pg_host="localhost",
        pg_database="postgres",
        pg_user="wmsusr",
        pg_password="pw",
        openai_api_key="sk-test",
        openai_embedding_model="text-embedding-3-small",
        unkey_root_api_key="unkey-test",
    )
    defaults.update(overrides)
    # _env_file=None: do not let a real .env on disk influence the test.
    return Settings(_env_file=None, **defaults)


def test_ssl_disabled_by_default_for_local_docker():
    settings = _make_settings(pg_ssl_mode="disable")
    store = settings.vector_store

    assert "sslmode=disable" in store.connection_string
    # asyncpg rejects ssl=disable as a literal value - the fix omits the
    # query param entirely rather than passing a value it can't parse.
    assert "ssl=" not in store.async_connection_string


def test_ssl_required_for_managed_postgres():
    settings = _make_settings(pg_ssl_mode="require")
    store = settings.vector_store

    assert "sslmode=require" in store.connection_string
    assert "ssl=require" in store.async_connection_string


def test_password_with_at_symbol_is_escaped_not_mangled():
    settings = _make_settings(pg_password="Ferrari123@")
    store = settings.vector_store

    # The real password must round-trip correctly (percent-encoded), and
    # must not be masked as "***" (SQLAlchemy's default URL.__str__
    # behaviour, which render_as_string(hide_password=False) exists to
    # avoid).
    assert "***" not in store.connection_string
    assert "Ferrari123%40" in store.connection_string


def test_table_and_schema_pass_through():
    settings = _make_settings(
        pg_table_name="custom_chunks", pg_schema_name="custom_schema"
    )
    store = settings.vector_store

    assert store.table_name == "custom_chunks"
    assert store.schema_name == "custom_schema"
