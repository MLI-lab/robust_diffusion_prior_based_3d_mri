import httpx
from prefect.context import get_run_context

PREFECT_API_URL = "http://127.0.0.1:4200/api"  # or https://api.prefect.cloud/api/accounts/... for Cloud

def get_flow_name_from_api():
    context = get_run_context()
    flow_id = context.flow_run.flow_id

    with httpx.Client() as client:
        response = client.get(f"{PREFECT_API_URL}/flows/{flow_id}")
        response.raise_for_status()
        flow_data = response.json()
        return flow_data["name"]