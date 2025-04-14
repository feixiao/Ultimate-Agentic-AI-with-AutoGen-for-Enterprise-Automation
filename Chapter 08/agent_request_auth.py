import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

# -----------------------------------------------------------------------------
# Constants and Logging Setup
# -----------------------------------------------------------------------------
# Define a session timeout for user authentication.
SESSION_TIMEOUT = timedelta(minutes=30)
# List of clinical systems that are authorized to make requests.
CLINICAL_SYSTEMS = ["EHR_MAIN", "CLINICAL_PORTAL", "PROVIDER_MOBILE"]

# Configure logger for authentication events.
logger = logging.getLogger("security.authentication")


# -----------------------------------------------------------------------------
# Enums and Classes
# -----------------------------------------------------------------------------
class AuthReason(Enum):
    """
    Enum representing possible reasons for an authentication result.
    """

    SUCCESS = "SUCCESS"
    INVALID_CERTIFICATE = "INVALID_CERTIFICATE"
    UNAUTHORIZED_ORIGIN = "UNAUTHORIZED_ORIGIN"
    UNAUTHORIZED_USER = "UNAUTHORIZED_USER"
    NO_TREATMENT_RELATIONSHIP = "NO_TREATMENT_RELATIONSHIP"


class AuthResult:
    """
    Class representing the result of an authentication process.
    """

    def __init__(
        self,
        authorized: bool,
        reason: AuthReason = AuthReason.SUCCESS,
        session_ttl: timedelta = SESSION_TIMEOUT,
    ):
        self.authorized = authorized
        self.reason = reason
        self.session_ttl = session_ttl


# -----------------------------------------------------------------------------
# Certificate Verification
# -----------------------------------------------------------------------------
def verify_system_certificate(certificate):
    """
    Verifies the digital signature of the agent system certificate.

    Checks:
      - Validity period of the certificate.
      - Issuer is among the trusted authorities.
      - (Simulated) Certificate revocation status.

    Args:
        certificate: A certificate object containing valid_from, valid_to, issuer, serial_number.

    Returns:
        bool: True if certificate is valid, otherwise False.
    """
    current_time = datetime.utcnow()
    # Validate certificate's active period.
    if not (certificate.valid_from <= current_time <= certificate.valid_to):
        return False

    # List of trusted certificate issuers.
    trusted_issuers = ["Healthcare CA", "Clinical Systems Authority"]
    if certificate.issuer not in trusted_issuers:
        return False

    # In a real system, check revocation status via CRL or OCSP.
    return True


# -----------------------------------------------------------------------------
# Origin Verification
# -----------------------------------------------------------------------------
def is_authorized_origin(origin, allowed_systems):
    """
    Verifies that the request originates from an authorized clinical system.

    Args:
        origin (str): The origin of the request.
        allowed_systems (list): List of authorized system identifiers.

    Returns:
        bool: True if the origin is authorized, otherwise False.
    """
    return origin in allowed_systems


# -----------------------------------------------------------------------------
# Clinician Context Retrieval
# -----------------------------------------------------------------------------
def get_clinician_context(session_id):
    """
    Retrieves clinician context based on the session identifier.

    In production, this would query a session store or identity provider.
    Here, a simulated clinician context is returned.

    Args:
        session_id (str): The session identifier.

    Returns:
        dict: Clinician details including id, authentication status, permissions, and last authentication time.
    """
    clinician = {
        "id": "PROV123456",
        "is_authenticated": True,
        "permissions": ["DOCUMENTATION_AGENT", "VIEW_CHARTS"],
        "last_authentication": datetime.utcnow() - timedelta(minutes=10),
    }

    # Invalidate the clinician session if the authentication is outdated.
    if (datetime.utcnow() - clinician["last_authentication"]) > SESSION_TIMEOUT:
        clinician["is_authenticated"] = False

    return clinician


# -----------------------------------------------------------------------------
# Permission Check
# -----------------------------------------------------------------------------
def has_permission(clinician, required_permission):
    """
    Checks if the clinician has the required permission.

    Args:
        clinician (dict): Clinician context containing permissions.
        required_permission (str): The required permission to check.

    Returns:
        bool: True if authenticated and permission is present, else False.
    """
    if not clinician["is_authenticated"]:
        return False
    return required_permission in clinician["permissions"]


# -----------------------------------------------------------------------------
# Patient ID Extraction
# -----------------------------------------------------------------------------
def extract_patient_id(request):
    """
    Extracts the patient ID from the request payload.

    Checks both top-level and nested contexts within the payload.

    Args:
        request: The request object containing a payload attribute.

    Returns:
        The patient ID if found, otherwise None.
    """
    if "patient_id" in request.payload:
        return request.payload["patient_id"]
    elif "context" in request.payload and "patient" in request.payload["context"]:
        return request.payload["context"]["patient"]
    return None


# -----------------------------------------------------------------------------
# Treatment Relationship Verification
# -----------------------------------------------------------------------------
def verify_treatment_relationship(clinician_id, patient_id):
    """
    Verifies if the clinician has a valid treatment relationship with the patient.

    In production, this would involve querying the EHR system.
    Here, a simulated list of valid relationships is used.

    Args:
        clinician_id (str): Identifier for the clinician.
        patient_id (str): Identifier for the patient.

    Returns:
        bool: True if the relationship is valid, else False.
    """
    valid_relationships = [("PROV123456", "PAT987654"), ("PROV123456", "PAT555555")]
    return (clinician_id, patient_id) in valid_relationships


# -----------------------------------------------------------------------------
# Main Authentication Function
# -----------------------------------------------------------------------------
def authenticate_agent_request(request, context):
    """
    Authenticates and authorizes AI agent requests in a healthcare setting.

    Authentication steps include:
      1. Verifying the system's digital certificate.
      2. Validating the request's origin.
      3. Checking the clinician's authentication status and permissions.
      4. Verifying the treatment relationship between clinician and patient (if applicable).

    Args:
        request: Object containing certificate, origin, and payload.
        context: Request context containing session information (e.g., session_id).

    Returns:
        AuthResult: The result of the authentication process.
    """
    # Step 1: Verify digital certificate.
    if not verify_system_certificate(request.certificate):
        logger.warning(
            f"Certificate validation failed: {request.certificate.serial_number}"
        )
        return AuthResult(authorized=False, reason=AuthReason.INVALID_CERTIFICATE)

    # Step 2: Validate request origin.
    if not is_authorized_origin(request.origin, CLINICAL_SYSTEMS):
        logger.warning(f"Unauthorized origin: {request.origin}")
        return AuthResult(authorized=False, reason=AuthReason.UNAUTHORIZED_ORIGIN)

    # Step 3: Retrieve clinician context and validate authentication and permissions.
    clinician = get_clinician_context(context.session_id)
    if not clinician["is_authenticated"]:
        logger.warning(f"Unauthorized user: {clinician['id']}")
        return AuthResult(authorized=False, reason=AuthReason.UNAUTHORIZED_USER)

    if not has_permission(clinician, "DOCUMENTATION_AGENT"):
        logger.warning(f"Missing required permission for user: {clinician['id']}")
        return AuthResult(authorized=False, reason=AuthReason.UNAUTHORIZED_USER)

    # Step 4: Verify treatment relationship if patient ID is provided.
    patient_id = extract_patient_id(request)
    if patient_id and not verify_treatment_relationship(clinician["id"], patient_id):
        logger.warning(
            f"No treatment relationship: Clinician {clinician['id']} to patient {patient_id}"
        )
        return AuthResult(authorized=False, reason=AuthReason.NO_TREATMENT_RELATIONSHIP)

    # All checks passed; authentication is successful.
    return AuthResult(authorized=True, session_ttl=SESSION_TIMEOUT)
