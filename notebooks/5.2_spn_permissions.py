# Databricks notebook source
# for the spn permissions locally instead of UI
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.iam import AccessControlRequest, PermissionLevel

from valuation_curator.config import ProjectConfig

cfg = ProjectConfig.from_yaml("../project_config.yml")
w = WorkspaceClient()

# COMMAND ----------
spn_app_id = dbutils.secrets.get("dev_SPN", "client_id")


# COMMAND ----------
w.permissions.update(
    request_object_type="genie",
    request_object_id=cfg.genie_space_id,
    access_control_list=[
        AccessControlRequest(
            service_principal_name=spn_app_id,
            permission_level=PermissionLevel.CAN_RUN,
        )
    ],
)

# COMMAND ----------
vs_endpoint = w.vector_search_endpoints.get_endpoint(cfg.vector_search_endpoint)
w.permissions.update(
    request_object_type="vector-search-endpoints",
    request_object_id=vs_endpoint.id,
    access_control_list=[
        AccessControlRequest(
            service_principal_name=spn_app_id,
            permission_level=PermissionLevel.CAN_USE,
        )
    ],
)

# COMMAND ----------
w.permissions.update(
    request_object_type="warehouses",
    request_object_id=cfg.warehouse_id,
    access_control_list=[
        AccessControlRequest(
            service_principal_name=spn_app_id,
            permission_level=PermissionLevel.CAN_USE,
        )
    ],
)
