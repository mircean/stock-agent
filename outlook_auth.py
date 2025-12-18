"""
Outlook OAuth2 authentication module.
"""

import logging
import os

import msal

logger = logging.getLogger(__name__)


class OutlookAuthenticator:
    """Handles OAuth2 authentication for Outlook API."""

    def __init__(self, tenant_id, client_id, scopes, cache_file="msal_cache.json"):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.scopes = scopes
        self.cache_file = cache_file
        self.authority = f"https://login.microsoftonline.com/{tenant_id}"

        self.cache = msal.SerializableTokenCache()
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                self.cache.deserialize(f.read())

        self.app = msal.PublicClientApplication(
            client_id=client_id,
            authority=self.authority,
            token_cache=self.cache
        )

    def _save_cache(self):
        """Save token cache to file if changed."""
        if self.cache.has_state_changed:
            with open(self.cache_file, 'w') as f:
                f.write(self.cache.serialize())

    def get_access_token(self):
        """Get access token using cached tokens or device flow."""
        # Try silent authentication first
        accounts = self.app.get_accounts()
        if accounts:
            result = self.app.acquire_token_silent(self.scopes, account=accounts[0])
            if result and "access_token" in result:
                self._save_cache()
                logger.debug("Silent authentication successful")
                return result["access_token"]

        # Fall back to device flow
        logger.info("No cached token found, initiating device flow authentication...")

        flow = self.app.initiate_device_flow(scopes=self.scopes)
        if "user_code" not in flow:
            raise SystemExit(flow.get("error_description", "Device flow failed"))

        print(f"\nSign in at {flow['verification_uri']} and enter code: {flow['user_code']}\n")

        result = self.app.acquire_token_by_device_flow(flow)

        if "access_token" not in result:
            raise SystemExit(result.get("error_description", "Authentication failed"))

        self._save_cache()
        logger.info("Authentication successful")
        return result["access_token"]