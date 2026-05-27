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

        Returns ``url`` and an ``invitations`` array. Received invitations
        include sender/target triage fields. Sent invitations include recipient
        identity and headline fields.
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
        title="Get Connections",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"network", "scraping"},
        exclude_args=["extractor"],
    )
    async def get_connections(
        ctx: Context,
        limit: Annotated[int, Field(ge=1, le=100)] = 20,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        List the authenticated user's most recently added 1st-degree connections.

        Returns ``url`` and a ``connections`` array. Each connection is
        ``{name, url, headline, connected_on}``. ``url`` is the relative
        ``/in/<slug>/`` profile path. ``connected_on`` is an ISO date
        (``YYYY-MM-DD``) parsed from the en-US "Connected on Month DD, YYYY"
        line, or ``None`` for other locales / unparseable text.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="get_connections"
            )
            logger.info("Fetching connections (limit=%d)", limit)
            await ctx.report_progress(
                progress=0, total=100, message="Loading connections"
            )
            result = await extractor.get_connections(limit=limit)
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_connections")
        except Exception as e:
            raise_tool_error(e, "get_connections")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Ignore Connection Request",
        annotations={"destructiveHint": True, "openWorldHint": True},
        tags={"network", "actions"},
        exclude_args=["extractor"],
    )
    async def ignore_connection_request(
        linkedin_username: str,
        ctx: Context,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Ignore a received LinkedIn connection request.

        Navigates to ``/in/{linkedin_username}/``, verifies the page is
        showing an incoming connection request, and clicks the Ignore
        button in the top-card action row. The Ignore button is
        identified via a locale-table label scan (per
        ``INCOMING_REQUEST_LABELS``) with structural fallbacks
        (engineering attrs → design-system class → documented position).

        Status: ``ignored`` | ``not_found`` | ``already_connected``
        | ``action_unavailable`` | ``verification_failed``.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="ignore_connection_request"
            )
            logger.info("Ignoring connection request: %s", linkedin_username)
            result = await extractor.act_on_invitation(linkedin_username, "ignore")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "ignore_connection_request")
        except Exception as e:
            raise_tool_error(e, "ignore_connection_request")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Withdraw Invitation",
        annotations={"destructiveHint": True, "openWorldHint": True},
        tags={"network", "actions"},
        exclude_args=["extractor"],
    )
    async def withdraw_invitation(
        linkedin_username: str,
        ctx: Context,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Withdraw an outgoing LinkedIn connection request.

        Navigates to ``/mynetwork/invitation-manager/sent/``, finds the card
        for ``linkedin_username``, and clicks Withdraw.

        Status: ``withdrawn`` | ``not_found`` | ``action_unavailable``
        | ``verification_failed``.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="withdraw_invitation"
            )
            logger.info("Withdrawing invitation: %s", linkedin_username)
            result = await extractor.act_on_invitation(linkedin_username, "withdraw")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "withdraw_invitation")
        except Exception as e:
            raise_tool_error(e, "withdraw_invitation")  # NoReturn
