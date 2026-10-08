"""What a provider receives: the lead as a provider would see it.

`lead_to_provider_message(lead)` is the hook a real dispatch step (SMS or email
to each matched provider) would call. Everything here is built from the `Lead`
only, never from internal state, so nothing like coordinates, search queries
or routing data can leak to a provider. The customer's last name is shortened
to an initial; the provider still gets the contact method to call back.
"""

from dataclasses import dataclass

from agent.categories import CATEGORIES
from agent.nodes import URGENCY_LABELS  # same wording the customer saw at confirmation
from agent.state import Lead


@dataclass(frozen=True)
class ProviderView:
    header: str  # "Emergency · Water damage restoration · 95616"
    description: str
    details: list[tuple[str, str]]  # (label, value), unknown answers left out
    customer: str  # "Jane D."
    contact: str  # "Phone: (530) 555-0123"
    extras: list[str]  # address, availability, owner/renter, safety notes (only when known)
    footer: str  # "Shared with 3 local pros · customer consented · ref abc123"


def short_name(name: str) -> str:
    """'Jane Doe' -> 'Jane D.'; a single name stays as is."""
    parts = name.split()
    return " ".join(parts) if len(parts) < 2 else f"{parts[0]} {parts[-1][0].upper()}."


def provider_view(lead: Lead) -> ProviderView:
    details = [
        (key.replace("_", " ").capitalize(), str(value))
        for key, value in lead.category_details.items()
        if value and str(value).lower() != "unknown"
    ]
    contact = " · ".join(
        f"{label}: {value}" for label, value in [("Phone", lead.contact_phone), ("Email", lead.contact_email)] if value
    )
    extras = [
        f"{label}: {value}"
        for label, value in [
            ("Address", lead.address),
            ("Availability", lead.availability),
            ("Owner or renter", lead.owner_or_renter),
            ("Safety", ", ".join(f.replace("_", " ") for f in lead.safety_flags)),
        ]
        if value
    ]
    n = len(lead.matched_providers)
    return ProviderView(
        header=f"{URGENCY_LABELS[lead.urgency]} · {CATEGORIES[lead.category].label} · {lead.zip}",
        description=lead.problem_description,
        details=details,
        customer=short_name(lead.name),
        contact=contact,
        extras=extras,
        footer=f"Shared with {n} local pro{'' if n == 1 else 's'} · customer consented · ref {lead.lead_id}",
    )


def lead_to_provider_message(lead: Lead) -> str:
    """Plain-text SMS/email body for one provider."""
    v = provider_view(lead)
    lines = [f"New lead: {v.header}", "", v.description, ""]
    lines += [f"{label}: {value}" for label, value in v.details]
    lines += [f"Customer: {v.customer}", v.contact, *v.extras, "", v.footer]
    return "\n".join(lines)
