import asyncio

async def delete_all():
    from agentcore_agent_runtime import delete_agent_runtime
    from agentcore_gateway import delete as del_gw
    from agentcore_memory import delete_agentcore_all_namespaces_records as del_mem
    from cognito_credentials_provider import delete as delete_cognito
    from google_credentials_provider import delete as del_google_cred
    
    print("Deleting Agent runtime...")
    agent_runtime_task = delete_agent_runtime()
    
    print("Deleting Agentcore GW...")
    gateway_task = del_gw()
    
    print("Deleting Agentcore memory...")
    memory_task = del_mem()
    
    print("Deleting cognito...")
    cognito_task = delete_cognito()
    
    print("Deleting google credentials...")
    google_task = del_google_cred()
    
    # Wait for all tasks to complete
    tasks = [
        agent_runtime_task,
        gateway_task,
        memory_task,
        cognito_task,
        google_task
    ]
    
    # Filter out any non-awaitable tasks (in case some functions don't return a coroutine)
    awaitable_tasks = [task for task in tasks if task is not None and hasattr(task, "__await__")]
    
    if awaitable_tasks:
        await asyncio.gather(*awaitable_tasks)
    
    print("All Agentcore deletion tasks completed")

# Run the async function
def main():
    asyncio.run(delete_all())