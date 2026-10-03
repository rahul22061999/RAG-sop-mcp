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
    return Settings(_env_file=None, **defaults)


def test_ssl_disabled_by_default_for_local_docker():
    settings = _make_settings(pg_ssl_mode="disable")
    store = settings.vector_store

    assert "sslmode=disable" in store.connection_string
    assert "ssl=" not in store.async_connection_string


def test_ssl_required_for_managed_postgres():
    settings = _make_settings(pg_ssl_mode="require")
    store = settings.vector_store

    assert "sslmode=require" in store.connection_string
    assert "ssl=require" in store.async_connection_string


def test_password_with_at_symbol_is_escaped_not_mangled():
    settings = _make_settings(pg_password="Ferrari123@")
    store = settings.vector_store

    assert "***" not in store.connection_string
    assert "Ferrari123%40" in store.connection_string


def test_table_and_schema_pass_through():
    settings = _make_settings(
        pg_table_name="custom_chunks", pg_schema_name="custom_schema"
    )
    store = settings.vector_store

    assert store.table_name == "custom_chunks"
    assert store.schema_name == "custom_schema"
