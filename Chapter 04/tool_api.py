from typing import Dict, Any, Optional
import requests
from enum import Enum
import json
import logging


# Enum for HTTP methods to enforce valid method types
class HttpMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


class APITool:
    def __init__(self, base_url: str, auth_token: Optional[str] = None):
        """
        Initialize the APITool instance.

        Parameters:
        * base_url: Base URL for API endpoints.
        * auth_token: Optional Bearer token for authentication.
        """
        # Normalize base URL by removing trailing slash
        self.base_url = base_url.rstrip("/")
        # Create a persistent session for HTTP requests
        self.session = requests.Session()

        # Set authentication header if auth token is provided
        if auth_token:
            self.session.headers.update({"Authorization": f"Bearer {auth_token}"})

        # Set default headers for JSON communication
        self.session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )

    def make_request(
        self,
        endpoint: str,
        method: HttpMethod,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request with the specified method and parameters.

        Parameters:
        * endpoint: API endpoint (appended to base_url).
        * method: HTTP method as an HttpMethod enum.
        * params: Query parameters for the request.
        * data: JSON payload for POST/PUT requests.
        * headers: Additional HTTP headers.

        Returns:
        * Dictionary with response data, status code, and headers.
        * In case of error, returns error details.
        """
        try:
            # Construct full URL by appending endpoint to base_url
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            request_kwargs = {"params": params or {}, "headers": headers or {}}

            # Include JSON payload for POST and PUT methods
            if method in [HttpMethod.POST, HttpMethod.PUT] and data:
                request_kwargs["json"] = data

            # Send the HTTP request using the session
            response = self.session.request(method.value, url, **request_kwargs)

            # Raise exception for HTTP error responses
            response.raise_for_status()

            # Attempt to parse response as JSON; fallback to text if JSON decoding fails
            try:
                return {
                    "status_code": response.status_code,
                    "data": response.json(),
                    "headers": dict(response.headers),
                }
            except json.JSONDecodeError:
                return {
                    "status_code": response.status_code,
                    "data": response.text,
                    "headers": dict(response.headers),
                }

        except requests.exceptions.RequestException as e:
            # Log error details for troubleshooting
            logging.error(f"API request failed: {str(e)}")
            return {
                "error": str(e),
                "status_code": getattr(e.response, "status_code", None)
                if hasattr(e, "response")
                else None,
            }

    def run(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an API query with validation and error handling.

        Parameters:
        * query: Dictionary containing at least 'endpoint' and 'method' keys.

        Returns:
        * Dictionary with the API response or error details.
        """
        try:
            # Validate that required fields exist in the query
            required_fields = ["endpoint", "method"]
            for field in required_fields:
                if field not in query:
                    raise ValueError(f"Missing required field: {field}")

            # Convert method string to HttpMethod enum; ensure uppercase conversion
            try:
                method = HttpMethod(query["method"].upper())
            except ValueError:
                raise ValueError(f"Invalid HTTP method: {query['method']}")

            # Execute the API request using the provided query parameters
            return self.make_request(
                endpoint=query["endpoint"],
                method=method,
                params=query.get("params"),
                data=query.get("data"),
                headers=query.get("headers"),
            )

        except Exception as e:
            # Log any exception that occurs during request processing
            logging.error(f"API tool error: {str(e)}")
            return {"error": str(e)}
