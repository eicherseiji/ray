HAPROXY_HEALTHZ_RULES_TEMPLATE = """    # Health check endpoint
    acl healthcheck path -i {{ config.health_check_endpoint }}
{%- if not health_info.healthy %}
    # Override: force health checks to fail (used by drain/disable)
    http-request return status {{ health_info.status }} content-type text/plain string "{{ health_info.health_message }}" if healthcheck
{%- elif backends %}
    # 200 if any backend has at least one server UP
{%-   for backend in backends %}
    acl backend_{{ backend.name or 'unknown' }}_server_up nbsrv({{ backend.name or 'unknown' }}) ge 1
{%-   endfor %}
    # Any backend with a server UP passes the health check (OR logic)
{%-   for backend in backends %}
    http-request return status {{ health_info.status }} content-type text/plain string "{{ health_info.health_message }}" if healthcheck backend_{{ backend.name or 'unknown' }}_server_up
{%-   endfor %}
    http-request return status 503 content-type text/plain string "Service Unavailable" if healthcheck
{%- endif %}
"""

HAPROXY_CONFIG_TEMPLATE = """global
    # Log to the standard system log socket with debug level.
    log /dev/log local0 debug
    stats socket {{ config.socket_path }} mode 666 level admin expose-fd listeners
    stats timeout 30s
    maxconn {{ config.maxconn }}
    nbthread {{ config.nbthread }}
defaults
    mode http
    option log-health-checks
    {% if config.timeout_connect_s is not none %}timeout connect {{ config.timeout_connect_s }}s{% endif %}
    {% if config.timeout_client_s is not none %}timeout client {{ config.timeout_client_s }}s{% endif %}
    {% if config.timeout_server_s is not none %}timeout server {{ config.timeout_server_s }}s{% endif %}
    {% if config.timeout_http_request_s is not none %}timeout http-request {{ config.timeout_http_request_s }}s{% endif %}
    {% if config.timeout_http_keep_alive_s is not none %}timeout http-keep-alive {{ config.timeout_http_keep_alive_s }}s{% endif %}
    {% if config.timeout_queue_s is not none %}timeout queue {{ config.timeout_queue_s }}s{% endif %}
    log global
    option httplog
    option abortonclose
    # Normalize 502 and 504 errors to 500 per Serve's default behavior
    {%- if config.error_file_path %}
    errorfile 502 {{ config.error_file_path }}
    errorfile 504 {{ config.error_file_path }}
    {%- endif %}
frontend prometheus
    bind :{{ config.metrics_port }}
    mode http
    http-request use-service prometheus-exporter if { path {{ config.metrics_uri }} }
    no log
frontend http_frontend
    bind {{ config.frontend_host }}:{{ config.frontend_port }}
{{ healthz_rules|safe }}
    # Routes endpoint
    acl routes path -i /-/routes
    http-request return status {{ route_info.status }} content-type {{ route_info.routes_content_type }} string "{{ route_info.routes_message }}" if routes

    {%- if config.inject_process_id_header and config.reload_id %}
    # Inject unique reload ID as header to track which HAProxy instance handled the request (testing only)
    http-request set-header x-haproxy-reload-id {{ config.reload_id }}
    {%- endif %}
    # Static routing based on path prefixes in decreasing length then alphabetical order
{%- for backend in backends %}
    acl is_{{ backend.name or 'unknown' }} path_beg {{ '/' if not backend.path_prefix or backend.path_prefix == '/' else backend.path_prefix ~ '/' }}
    acl is_{{ backend.name or 'unknown' }} path {{ backend.path_prefix or '/' }}
    use_backend {{ backend.name or 'unknown' }} if is_{{ backend.name or 'unknown' }}
{%- endfor %}
    default_backend default_backend
backend default_backend
    http-request return status 404 content-type text/plain lf-string "Path \'%[path]\' not found. Ping http://.../-/routes for available routes."
{%- for backend in backends %}
backend {{ backend.name or 'unknown' }}
    log global
    balance leastconn
    # Enable HTTP connection reuse for better performance
    http-reuse always
    # Set backend-specific timeouts, overriding defaults if specified
    {%- if backend.timeout_connect_s is not none %}
    timeout connect {{ backend.timeout_connect_s }}s
    {%- endif %}
    {%- if backend.timeout_server_s is not none %}
    timeout server {{ backend.timeout_server_s }}s
    {%- endif %}
    {%- if backend.timeout_client_s is not none %}
    timeout client {{ backend.timeout_client_s }}s
    {%- endif %}
    {%- if backend.timeout_http_request_s is not none %}
    timeout http-request {{ backend.timeout_http_request_s }}s
    {%- endif %}
    {%- if backend.timeout_queue_s is not none %}
    timeout queue {{ backend.timeout_queue_s }}s
    {%- endif %}
    # Set timeouts to support keep-alive connections
    {%- if backend.timeout_http_keep_alive_s is not none %}
    timeout http-keep-alive {{ backend.timeout_http_keep_alive_s }}s
    {%- endif %}
    {%- if backend.timeout_tunnel_s is not none %}
    timeout tunnel {{ backend.timeout_tunnel_s }}s
    {%- endif %}
    # Health check configuration - use backend-specific or global defaults
    {%- set fall_param = backend.health_check_fall if backend.health_check_fall is not none else config.health_check_fall -%}
    {%- set rise_param = backend.health_check_rise if backend.health_check_rise is not none else config.health_check_rise -%}
    {%- set inter_param = backend.health_check_inter if backend.health_check_inter is not none else config.health_check_inter -%}
    {%- set health_path = backend.health_check_path if backend.health_check_path is not none else config.health_check_path -%}
    {%- if health_path %}
    # HTTP health check with custom path
    option httpchk GET {{ health_path }}
    http-check expect status 200
    {%- endif %}
    {%- if fall_param is not none and rise_param is not none and inter_param is not none %}
    default-server fall {{ fall_param }} rise {{ rise_param }} inter {{ inter_param }} check
    {%- elif fall_param is not none or rise_param is not none or inter_param is not none %}
    default-server{% if fall_param is not none %} fall {{ fall_param }}{% endif %}{% if rise_param is not none %} rise {{ rise_param }}{% endif %}{% if inter_param is not none %} inter {{ inter_param }}{% endif %} check
    {%- else %}
    default-server check
    {%- endif %}
    # Servers in this backend
    {%- for server in backend.servers %}
    server {{ server.name }} {{ server.host }}:{{ server.port }} check
    {%- endfor %}
{%- endfor %}
listen stats
  bind *:{{ config.stats_port }}
  stats enable
  stats uri {{ config.stats_uri }}
  stats refresh 1s
"""
