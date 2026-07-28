"""
LinkedIn messaging tools.

Provides inbox listing, conversation reading, message search, and sending.
"""

import logging
from typing import Annotated, Any

from fastmcp import Context, FastMCP
from pydantic import Field

from linkedin_mcp_server.config.schema import DEFAULT_TOOL_TIMEOUT_SECONDS
from linkedin_mcp_server.core.exceptions import (
    AuthenticationError,
    LinkedInScraperException,
)
from linkedin_mcp_server.dependencies import get_ready_extractor, handle_auth_error
from linkedin_mcp_server.error_handler import raise_tool_error

logger = logging.getLogger(__name__)


def register_messaging_tools(
    mcp: FastMCP, *, tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS
) -> None:
    """Register all messaging-related tools with the MCP server."""

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Inbox",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"messaging", "scraping"},
        exclude_args=["extractor"],
    )
    async def get_inbox(
        ctx: Context,
        limit: Annotated[int, Field(ge=1, le=50)] = 20,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        List recent conversations from the LinkedIn messaging inbox.

        Args:
            ctx: FastMCP context for progress reporting
            limit: Maximum conversations to load per inbox tab (1-50, default 20)

        Returns:
            Dict with url, sections (inbox -> raw text), and optional references.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="get_inbox"
            )
            logger.info("Fetching inbox (limit=%d)", limit)

            await ctx.report_progress(
                progress=0, total=100, message="Loading messaging inbox"
            )

            result = await extractor.get_inbox(limit=limit)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_inbox")
        except Exception as e:
            raise_tool_error(e, "get_inbox")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Conversation",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"messaging", "scraping"},
        exclude_args=["extractor"],
    )
    async def get_conversation(
        ctx: Context,
        linkedin_username: str | None = None,
        thread_id: str | None = None,
        message_url: str | None = None,
        index: Annotated[int, Field(ge=0)] = 0,
        max_scrolls: Annotated[int, Field(ge=0, le=20)] = 3,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Read a specific messaging conversation as a structured list of messages.

        Provide a thread_id, or linkedin_username for inbox lookup. For a
        pending invitation sender who is not a direct connection, provide
        linkedin_username plus the message_url from get_pending_invitations.

        When looked up by linkedin_username, resolution searches the messaging
        inbox for the participant's display name and click-visits every
        matching row to capture its thread ID — LinkedIn's sidebar has no
        anchor hrefs or thread-id attributes, so this is the only available
        path. Each visit selects the row in the LinkedIn UI and may mark it
        as read. Pass thread_id directly to skip this enumeration.

        Args:
            ctx: FastMCP context for progress reporting
            linkedin_username: LinkedIn username of the conversation participant
            thread_id: LinkedIn messaging thread ID
            message_url: Relative invitation compose URL from
                get_pending_invitations. Requires linkedin_username and cannot
                be combined with thread_id.
            index: 0-based selector for which thread to open when the
                participant has multiple threads (e.g. an organic 1-on-1 plus
                an InMail). Ignored when thread_id is provided. To enumerate
                thread IDs first, call search_conversations.
            max_scrolls: How many times to scroll the message list back to
                load older history (0-20, default 3). LinkedIn virtualizes
                the list, so message count grows with scroll count; raise
                this value to retrieve older history at the cost of latency.

        Returns:
            Dict with url and a sections payload of the form
            ``{"messages": [...], "members": [...]}``. Each message is
            ``{timestamp, status, sender, content}``; status is one of
            ``sent`` / ``read`` / ``delivered`` / ``deleted`` (only
            ``sent`` and ``deleted`` are reliably emitted today). Timestamp
            and deleted-status parsing are en-US best-effort.
        """
        if not linkedin_username and not thread_id and not message_url:
            raise_tool_error(
                LinkedInScraperException(
                    "Provide at least one of linkedin_username or thread_id"
                ),
                "get_conversation",
            )
        if message_url and not linkedin_username:
            raise_tool_error(
                LinkedInScraperException(
                    "linkedin_username is required with message_url"
                ),
                "get_conversation",
            )
        if message_url and thread_id:
            raise_tool_error(
                LinkedInScraperException(
                    "message_url cannot be combined with thread_id"
                ),
                "get_conversation",
            )

        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="get_conversation"
            )
            logger.info(
                "Fetching conversation: username=%s, thread_id=%s, "
                "invitation=%s, index=%d, max_scrolls=%d",
                linkedin_username,
                thread_id,
                message_url is not None,
                index,
                max_scrolls,
            )

            await ctx.report_progress(
                progress=0, total=100, message="Loading conversation"
            )

            result = await extractor.get_conversation(
                linkedin_username=linkedin_username,
                thread_id=thread_id,
                message_url=message_url,
                index=index,
                max_scrolls=max_scrolls,
            )

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_conversation")
        except Exception as e:
            raise_tool_error(e, "get_conversation")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Archive Conversation",
        annotations={"destructiveHint": True, "openWorldHint": True},
        tags={"messaging", "actions"},
        exclude_args=["extractor"],
    )
    async def archive_conversation(
        ctx: Context,
        linkedin_username: str | None = None,
        thread_id: str | None = None,
        message_url: str | None = None,
        index: Annotated[int, Field(ge=0)] = 0,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """Archive a messaging conversation.

        Identify it by ``thread_id`` or ``linkedin_username``. For a pending
        invitation conversation, provide ``linkedin_username`` plus the
        ``message_url`` returned by ``get_pending_invitations``.
        """
        if not linkedin_username and not thread_id and not message_url:
            raise_tool_error(
                LinkedInScraperException(
                    "Provide at least one of linkedin_username or thread_id"
                ),
                "archive_conversation",
            )
        if message_url and not linkedin_username:
            raise_tool_error(
                LinkedInScraperException(
                    "linkedin_username is required with message_url"
                ),
                "archive_conversation",
            )
        if message_url and thread_id:
            raise_tool_error(
                LinkedInScraperException(
                    "message_url cannot be combined with thread_id"
                ),
                "archive_conversation",
            )

        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="archive_conversation"
            )
            logger.info(
                "Archiving conversation: username=%s, thread_id=%s, invitation=%s",
                linkedin_username,
                thread_id,
                message_url is not None,
            )
            return await extractor.archive_conversation(
                linkedin_username=linkedin_username,
                thread_id=thread_id,
                message_url=message_url,
                index=index,
            )
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "archive_conversation")
        except Exception as e:
            raise_tool_error(e, "archive_conversation")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Search Conversations",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"messaging", "search"},
        exclude_args=["extractor"],
    )
    async def search_conversations(
        keywords: str,
        ctx: Context,
        limit: Annotated[int, Field(ge=1, le=50)] = 20,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Search messages by keyword.

        Args:
            keywords: Search keywords to filter conversations
            ctx: FastMCP context for progress reporting
            limit: Maximum number of search-result rows to enumerate as
                conversation references (1-50, default 20). Each enumeration
                selects the row in LinkedIn's UI and may mark it as read, so
                a low cap is preferable for noisy queries.

        Returns:
            Dict with url, sections (search_results -> raw text), and optional references.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="search_conversations"
            )
            logger.info(
                "Searching conversations: keywords='%s', limit=%d", keywords, limit
            )

            await ctx.report_progress(
                progress=0, total=100, message="Searching messages"
            )

            result = await extractor.search_conversations(keywords, limit=limit)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "search_conversations")
        except Exception as e:
            raise_tool_error(e, "search_conversations")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Send Message",
        annotations={"destructiveHint": True, "openWorldHint": True},
        tags={"messaging", "actions"},
        exclude_args=["extractor"],
    )
    async def send_message(
        linkedin_username: str,
        message: str,
        confirm_send: bool,
        ctx: Context,
        profile_urn: str | None = None,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Send a message to a LinkedIn user.

        The recipient must be directly messageable from the profile page. This is a
        write operation when confirm_send is True.

        Args:
            linkedin_username: LinkedIn username of the recipient
            message: The message text to send. `\n` is preserved as a line
                break and `\n\n` as a paragraph break, matching how LinkedIn
                renders multi-paragraph messages in the web composer.
            confirm_send: Must be True to send the message
            ctx: FastMCP context for progress reporting
            profile_urn: Optional profile URN (e.g. ACoAAB...) to construct the
                compose URL directly. Providing this bypasses the Message-button
                lookup and is more reliable when available. Obtain via
                get_person_profile. Note: inbox may not always show all
                messages; use search_conversations as a fallback.

        Returns:
            Dict with url, status, message, recipient_selected, and sent.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="send_message"
            )
            logger.info(
                "Sending message to %s (confirm_send=%s)",
                linkedin_username,
                confirm_send,
            )

            await ctx.report_progress(progress=0, total=100, message="Sending message")

            result = await extractor.send_message(
                linkedin_username,
                message,
                confirm_send=confirm_send,
                profile_urn=profile_urn,
            )

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "send_message")
        except Exception as e:
            raise_tool_error(e, "send_message")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Message Invitation Sender",
        annotations={"destructiveHint": True, "openWorldHint": True},
        tags={"messaging", "actions"},
        exclude_args=["extractor"],
    )
    async def message_invitation_sender(
        linkedin_username: str,
        message_url: str,
        message: str,
        confirm_send: bool,
        ctx: Context,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """Message the sender of a received connection invitation.

        ``message_url`` must be the relative compose URL returned by
        ``get_pending_invitations`` for the sender. Set ``confirm_send`` to
        ``True`` to send; ``False`` performs a recipient-verified dry run.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="message_invitation_sender"
            )
            logger.info(
                "Messaging invitation sender %s (confirm_send=%s)",
                linkedin_username,
                confirm_send,
            )
            return await extractor.send_message(
                linkedin_username,
                message,
                confirm_send=confirm_send,
                compose_url=message_url,
            )
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "message_invitation_sender")
        except Exception as e:
            raise_tool_error(e, "message_invitation_sender")  # NoReturn
