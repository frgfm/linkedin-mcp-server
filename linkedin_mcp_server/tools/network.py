"""LinkedIn network invitation tools."""

import logging
from typing import Annotated, Any, Literal

from fastmcp import Context, FastMCP
from pydantic import Field

from linkedin_mcp_server.config.schema import DEFAULT_TOOL_TIMEOUT_SECONDS
from linkedin_mcp_server.core.exceptions import AuthenticationError
from linkedin_mcp_server.dependencies import get_ready_extractor, handle_auth_error
from linkedin_mcp_server.error_handler import raise_tool_error

logger = logging.getLogger(__name__)


def register_network_tools(
    mcp: FastMCP, *, tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS
) -> None:
    """Register network invitation tools with the MCP server."""

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Pending Invitations",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"network", "scraping"},
        exclude_args=["extractor"],
    )
    async def get_pending_invitations(
        ctx: Context,
        limit: Annotated[int, Field(ge=1, le=100)] = 20,
        kind: Literal["received", "sent"] = "received",
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        List pending LinkedIn network invitations (received or sent).

        Returns the standard scraping shape plus an optional compact
        ``invitations`` array with profile identity for follow-up actions.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="get_pending_invitations"
            )
            logger.info("Fetching pending invitations (kind=%s, limit=%d)", kind, limit)
            await ctx.report_progress(
                progress=0, total=100, message=f"Loading {kind} invitations"
            )
            result = await extractor.get_pending_invitations(limit=limit, kind=kind)
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_pending_invitations")
        except Exception as e:
            raise_tool_error(e, "get_pending_invitations")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Accept Invitation",
        annotations={"destructiveHint": True, "openWorldHint": True},
        tags={"network", "actions"},
        exclude_args=["extractor"],
    )
    async def accept_invitation(
        linkedin_username: str,
        confirm_accept: bool,
        ctx: Context,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """Accept a received LinkedIn invitation after explicit confirmation."""
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="accept_invitation"
            )
            logger.info(
                "Accepting invitation from %s (confirm_accept=%s)",
                linkedin_username,
                confirm_accept,
            )
            await ctx.report_progress(
                progress=0, total=100, message="Accepting invitation"
            )
            result = await extractor.accept_invitation(
                linkedin_username,
                confirm_accept=confirm_accept,
            )
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "accept_invitation")
        except Exception as e:
            raise_tool_error(e, "accept_invitation")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Reject Invitation",
        annotations={"destructiveHint": True, "openWorldHint": True},
        tags={"network", "actions"},
        exclude_args=["extractor"],
    )
    async def reject_invitation(
        linkedin_username: str,
        confirm_reject: bool,
        ctx: Context,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """Reject a received LinkedIn invitation after explicit confirmation."""
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="reject_invitation"
            )
            logger.info(
                "Rejecting invitation from %s (confirm_reject=%s)",
                linkedin_username,
                confirm_reject,
            )
            await ctx.report_progress(
                progress=0, total=100, message="Rejecting invitation"
            )
            result = await extractor.reject_invitation(
                linkedin_username,
                confirm_reject=confirm_reject,
            )
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "reject_invitation")
        except Exception as e:
            raise_tool_error(e, "reject_invitation")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Withdraw Invitation",
        annotations={"destructiveHint": True, "openWorldHint": True},
        tags={"network", "actions"},
        exclude_args=["extractor"],
    )
    async def withdraw_invitation(
        linkedin_username: str,
        confirm_withdraw: bool,
        ctx: Context,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """Withdraw a sent LinkedIn invitation after explicit confirmation."""
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="withdraw_invitation"
            )
            logger.info(
                "Withdrawing invitation to %s (confirm_withdraw=%s)",
                linkedin_username,
                confirm_withdraw,
            )
            await ctx.report_progress(
                progress=0, total=100, message="Withdrawing invitation"
            )
            result = await extractor.withdraw_invitation(
                linkedin_username,
                confirm_withdraw=confirm_withdraw,
            )
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "withdraw_invitation")
        except Exception as e:
            raise_tool_error(e, "withdraw_invitation")  # NoReturn
