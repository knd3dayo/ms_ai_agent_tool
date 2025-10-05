import os, json, argparse
from typing import Any, Union, ClassVar, Optional, Any, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

from agent_framework import MCPStdioTool, HostedMCPTool

import ms_ai_agent_tool.log_modules.log_settings as log_settings
logger = log_settings.getLogger(__name__)


class MSAIAgentProps(BaseModel):
    openai_key: str = Field(default="", alias="openai_key")
    azure_openai: bool = Field(default=False, alias="azure_openai")
    azure_openai_api_version: Optional[str] = Field(default=None, alias="azure_openai_api_version")
    azure_openai_endpoint: Optional[str] = Field(default=None, alias="azure_openai_endpoint")
    openai_base_url: Optional[str] = Field(default=None, alias="openai_base_url")

    default_completion_model: str = Field(default="gpt-4o", alias="default_completion_model")
    default_embedding_model: str = Field(default="text-embedding-3-small", alias="default_embedding_model")

    @model_validator(mode='before')
    def handle_azure_openai_bool_and_version(cls, values):
        azure_openai = values.get("azure_openai", False)
        if isinstance(azure_openai, str):
            values["azure_openai"] = azure_openai.upper() == "TRUE"
        if values.get("azure_openai_api_version") is None:
            values["azure_openai_api_version"] = "2024-02-01"
        return values


    def create_openai_dict(self) -> dict:
        completion_dict = {}
        completion_dict["api_key"] = self.openai_key
        completion_dict["model_id"] = self.default_completion_model
        if self.openai_base_url:
            completion_dict["base_url"] = self.openai_base_url
        return completion_dict

    def create_azure_openai_dict(self) -> dict:
        completion_dict = {}
        completion_dict["api_key"] = self.openai_key
        if self.openai_base_url:
            completion_dict["base_url"] = self.openai_base_url
        else:
            completion_dict["endpoint"] = self.azure_openai_endpoint
            completion_dict["deployment_name"] = self.default_completion_model
            completion_dict["api_version"] = self.azure_openai_api_version
        return completion_dict

    @staticmethod
    def create_from_env() -> 'MSAIAgentProps':
        load_dotenv()
        props: dict = {
            "openai_key": os.getenv("OPENAI_API_KEY"),
            "azure_openai": os.getenv("AZURE_OPENAI"),
            "azure_openai_api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
            "azure_openai_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "openai_base_url": os.getenv("OPENAI_BASE_URL"),
            "default_completion_model": os.getenv("OPENAI_COMPLETION_MODEL", "gpt-4o"),
            "default_embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        }
        openAIProps = MSAIAgentProps.model_validate(props)
        return openAIProps



class MSAIAgentMcpSetting(BaseModel):
    # mcp_settings.jsonのルートとなる要素名
    servers_label: ClassVar[str] = "mcpServers"

    # typeはstdio or sse. defaultはstdio
    type: str = Field(default="stdio", description="Type of the MCP tool (e.g., 'stdio', 'sse').")
    name: str = Field(description="Name of the MCP tool.")
    autoApprove: Optional[list[str]] = Field(default=[], description="List of tool names allowed to auto-approve the tool execution.")
    disabled: Optional[bool] = Field(default=False, description="Whether the tool is disabled.")
    description: Optional[str] = Field(default=None, description="Description of the MCP tool.")
    timeout: Optional[int] = Field(default=60, description="Timeout in seconds for the MCP tool execution.")

    # for stdio
    command: Optional[str] = Field(default=None, description="Command to execute for the MCP tool.")
    args: Optional[List[str]] = Field(default=None, description="Arguments for the MCP tool command.")
    env: Optional[dict] = Field(default=None, description="Environment variables for the MCP tool.")
    # for sse
    url: Optional[str] = Field(default=None, description="URL for the MCP tool if type is 'sse'.")


    @staticmethod
    def create_from_file(file_path: str) -> dict[str, 'MSAIAgentMcpSetting']:
        try:
            settings_dict: dict[str, MSAIAgentMcpSetting] = {}
            with open(file_path, 'r') as file:
                data = json.load(file)
                tools_data = data.get(MSAIAgentMcpSetting.servers_label, {})
                for tool_name, tool in tools_data.items():
                    tool["name"] = tool_name
                    validated_tool = MSAIAgentMcpSetting.model_validate(tool)
                    settings_dict[validated_tool.name] = validated_tool
            return settings_dict
        except Exception as e:
            logger.error(f"Error loading MCP settings from {file_path}: {e}")
            return {}

    @staticmethod
    def create_mcp_tools_from_settings(mcp_settings_json_path: str) -> list[Any]:
        if not mcp_settings_json_path or os.path.isfile(mcp_settings_json_path) is False:
            logger.info("MCP settings JSON path is not provided or invalid.")
            return []
        mcp_settings = MSAIAgentMcpSetting.create_from_file(mcp_settings_json_path)
        tools = []
        for name, setting in mcp_settings.items():
            if setting.disabled:
                logger.info(f"MCP tool '{name}' is disabled. Skipping.")
                continue
            if setting.type == "stdio":
                if not setting.command:
                    logger.error(f"MCP tool '{name}' of type 'stdio' requires a command. Skipping.")
                    continue
                tool = MCPStdioTool(
                    name=setting.name,
                    command=setting.command,
                    args=setting.args,
                    env=setting.env,
                    auto_approve=setting.autoApprove,
                    description=setting.description,
                    timeout=setting.timeout
                )
                tools.append(tool)
                logger.debug(f"Created MCPStdioTool: {setting.name}")
                
            elif setting.type == "sse":
                if not setting.url:
                    logger.error(f"MCP tool '{setting.name}' of type 'sse' requires a URL. Skipping.")
                    continue
                tool = HostedMCPTool(
                    name=setting.name,
                    url=setting.url,
                    auto_approve=setting.autoApprove,
                    description=setting.description,
                    timeout=setting.timeout
                )
                tools.append(tool)
                logger.debug(f"Created MCPSseTool: {setting.name}")
            else:
                logger.error(f"Unknown MCP tool type '{setting.type}' for tool '{setting.name}'. Skipping.")

        return tools


from agent_framework.azure import AzureOpenAIResponsesClient
from agent_framework.openai import OpenAIChatClient

class MSAIAgentUtil:
    def __init__(self, props: MSAIAgentProps):
        
        self.props = props

    def create_default_mcp_server(self) -> MCPStdioTool:
        tool = MCPStdioTool(
            name="mcp_server",
            command="uv",
            args=["run", "-m", "ms_ai_agent_tool.mcp_modules.mcp_server"],
            env=os.environ.copy(),
            description="MCP server tools for file operations",
        )
        return tool

    def create_client(self) -> Union[OpenAIChatClient, AzureOpenAIResponsesClient]:
        
        if (self.props.azure_openai):
            params = self.props.create_azure_openai_dict()
            agent = AzureOpenAIResponsesClient(
                **params
            )
            return agent

        else:
            params = self.props.create_openai_dict()
            agent = OpenAIChatClient(
                **params
            )
            return agent
        
    def create_instractions(self, custom_instructions_path: Optional[str] = None) -> str:
        logger.debug(f"Creating instructions with custom_instructions_path: {custom_instructions_path}")
        instructions = "You are a helpful assistant."
        if custom_instructions_path and os.path.isfile(custom_instructions_path):
            try:
                with open(custom_instructions_path, 'r', encoding='utf-8') as file:
                    custom_instructions = file.read()
                    if custom_instructions.strip():
                        # カスタムインストラクションの内容に従うことを明示的に指示
                        instructions = f"""
                            ユーザーからの指示を実行する前に必ず以下のカスタムインストラクションを確認し、遵守してください。
                            カスタムインストラクションの内容とユーザーの指示を踏まえて、最適な計画を考えてください。
                            回答する際には、回答がどのような計画に従って行われたかを説明してください。
                            カスタムインストラクションの内容は以下の通りです。
                            -----
                            {custom_instructions}
                            -----
                            上記のカスタムインストラクションに従ってください。
                            """
                        logger.info(f"Loaded custom instructions from {custom_instructions_path}")
                    else:
                        logger.warning(f"Custom instructions file {custom_instructions_path} is empty. Using default instructions.")
            except Exception as e:
                logger.error(f"Error reading custom instructions from {custom_instructions_path}: {e}. Using default instructions.")
        else:
            if custom_instructions_path:
                logger.warning(f"Custom instructions path {custom_instructions_path} is invalid. Using default instructions.")
        logger.debug(f"Final instructions: {instructions}")
        return instructions

async def async_main():
    # 引数解析 -f mcp_settings_json_path
    parser = argparse.ArgumentParser(description="MS AI Agent Sample")
    parser.add_argument("-f", "--mcp_settings_json_path", type=str, help="Path to the MCP settings JSON file")
    # -d 作業ディレクトリ defaultはカレントディレクトリ
    parser.add_argument("-d", "--working_directory", type=str, default=".", help="Path to the working directory")
    # 作業フォルダ以外のファイル更新を許可するかどうかのフラグ
    parser.add_argument("--allow_outside_modifications", action="store_true", help="Allow modifications to files outside the working directory")
    # カスタムインストラクションのパス
    parser.add_argument("-c", "--custom_instructions_path", type=str, help="Path to the custom instructions file")

    args = parser.parse_args()
    mcp_settings_json_path = args.mcp_settings_json_path
    working_directory = args.working_directory
    allow_outside_modifications = args.allow_outside_modifications
    custom_instructions_path = args.custom_instructions_path
    
    if working_directory and os.path.isdir(working_directory):
        os.chdir(working_directory)
        logger.info(f"Changed working directory to: {working_directory}")
    else:
        logger.warning(f"Working directory '{working_directory}' is invalid. Using current directory.")

    if allow_outside_modifications:
        os.environ["ALLOW_OUTSIDE_MODIFICATIONS"] = "true"
        logger.info("Modifications to files outside the working directory are allowed.")
    else:
        os.environ["ALLOW_OUTSIDE_MODIFICATIONS"] = "false"
        logger.info("Modifications to files outside the working directory are NOT allowed.")

    # Create an agent using OpenAI ChatCompletion
    agent_util = MSAIAgentUtil(MSAIAgentProps.create_from_env())
    client = agent_util.create_client()
    mcp_tools = MSAIAgentMcpSetting.create_mcp_tools_from_settings(mcp_settings_json_path)

    tools = [agent_util.create_default_mcp_server()] + mcp_tools

    params = {}
    params["name"] = "HelpfulAssistant"
    params["instructions"] = agent_util.create_instractions(custom_instructions_path)
    if tools:
        params["tools"] = tools

    async with (client.create_agent(
            **params
        ) as agent):
        # Create a thread for persistent conversation
        thread = agent.get_new_thread()
        while True:
            try:
                input_text = input("Enter your request: ")
                result = await agent.run(input_text, thread=thread)
                print(result)
            except InterruptedError as e:
                print("Interrupted. Exiting...")
                break


if __name__ == "__main__":
    import asyncio
    asyncio.run(async_main())
