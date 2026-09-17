"""Offline check of the installed MCP and real CLI; never contacts an instance."""
import asyncio
import json
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import MCPError


async def main():
    root = Path(__file__).resolve().parents[2]
    params = StdioServerParameters(
        command="/bin/bash",
        args=[str(root / "tools/autodl/start-mcp.sh")],
        cwd=str(root),
    )
    checks = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            assert init.server_info.name == "autodl-remote"
            checks.append("MCP SDK initialize with real installed server")
            catalog = await session.list_tools()
            names = [tool.name for tool in catalog.tools]
            assert len(names) == 13
            checks.append("13 tool schemas discovered")
            accounts = await session.call_tool("autodl_account", {"action": "list"})
            assert not accounts.is_error
            assert accounts.structured_content["exitCode"] == 0
            checks.append("MCP -> real CLI account list succeeds")
            for name, args, expected in [
                ("autodl_transfer", {"action": "upload", "projectRoot": str(root),
                                     "localPath": "../outside"}, "cannot escape projectRoot"),
                ("autodl_transfer", {"action": "upload", "projectRoot": str(root),
                                     "localPath": "tools/autodl/README.md", "delete": True}, "confirmDelete"),
            ]:
                try:
                    await session.call_tool(name, args)
                except MCPError as error:
                    assert expected in str(error), str(error)
                else:
                    raise AssertionError(f"Expected rejection: {expected}")
            checks.append("path traversal and unconfirmed mirror deletion rejected")
            if not (root / ".autodl-remote.conf").exists():
                result = await session.call_tool("autodl_doctor", {"projectRoot": str(root)})
                assert result.is_error
                assert result.structured_content["exitCode"] != 0
                checks.append("unbound project reports failure without SSH")
    report = {"status": "passed", "checks": checks, "tools": names,
              "remote_verified": False, "gpu_training_started": False}
    output = root / "tools/autodl/runtime/mcp-smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
