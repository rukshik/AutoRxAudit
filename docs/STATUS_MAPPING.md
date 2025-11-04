# Prescription Status Mapping Between Doctor and Pharmacy Systems

## Overview
The doctor and pharmacy databases use different status naming conventions. This document maps statuses between the two systems and defines the workflow states.

## Status Mapping Table

| Doctor System (doctor_office) | Pharmacy System (autorxaudit) | Description | Who Controls |
|-------------------------------|-------------------------------|-------------|--------------|
| `draft` | N/A | Prescription being created | Doctor |
| `sent_to_pharmacy` | `RECEIVED` | Prescription sent, awaiting AI | Doctor → Pharmacy |
| `sent_to_pharmacy` | `AI_APPROVED` | AI approved, awaiting pharmacist | Pharmacy (AI) |
| `sent_to_pharmacy` | `AI_FLAGGED` | AI flagged, awaiting pharmacist | Pharmacy (AI) |
| `sent_to_pharmacy` | `AI_ERROR` | AI processing error | Pharmacy (AI) |
| `pending_review` | `PENDING_REVIEW` | Awaiting pharmacist review | Pharmacy |
| `under_review` | `UNDER_REVIEW` | Pharmacist requested more info | Pharmacist |
| `pharmacy_approved` | `APPROVED` | Pharmacist approved | Pharmacist |
| `pharmacy_denied` | `DENIED` | Pharmacist denied | Pharmacist |
| `cancelled` | `CANCELLED` | Doctor cancelled | Doctor |

## Workflow States

### Phase 1: Creation & Transmission
```
Doctor System                 Pharmacy System
-------------                 ---------------
draft                         (not yet sent)
    ↓
sent_to_pharmacy  ------>     RECEIVED
```

### Phase 2: AI Processing
```
Doctor System                 Pharmacy System
-------------                 ---------------
sent_to_pharmacy              RECEIVED
                                  ↓
                             [AI Processing]
                                  ↓
sent_to_pharmacy         AI_APPROVED | AI_FLAGGED | AI_ERROR
```

### Phase 3: Pharmacist Review (New Workflow)
```
Doctor System                 Pharmacy System
-------------                 ---------------
pending_review    <----->     PENDING_REVIEW
                              (Pharmacist sees AI results)
                                      ↓
                              ┌──────┴──────┬──────────┐
                              ↓             ↓          ↓
pharmacy_approved <----- APPROVED      DENIED --->  pharmacy_denied
                              ↓
                         UNDER_REVIEW -----> under_review
                              ↓                    ↓
                         (Pharmacist          (Doctor responds
                          requests             or cancels)
                          review)                  ↓
                                              UNDER_REVIEW
                                              (back to pharmacist)
```

### Phase 4: Doctor Response to Review
```
Doctor System                 Pharmacy System
-------------                 ---------------
under_review      <----->     UNDER_REVIEW
    ↓                              ↑
    └─ Doctor responds ────────────┘
    └─ Doctor cancels ──────> CANCELLED -----> cancelled
```

## Communication Actions and Status Transitions

### Action: PHARMACIST_REQUEST_REVIEW
- **Pharmacy Status**: AI_FLAGGED/AI_APPROVED → UNDER_REVIEW
- **Doctor Status**: sent_to_pharmacy → under_review
- **Actor**: Pharmacist
- **Blockchain Event**: PharmacistRequestsReview

### Action: DOCTOR_RESPONSE
- **Pharmacy Status**: UNDER_REVIEW (stays)
- **Doctor Status**: under_review (stays)
- **Actor**: Doctor
- **Blockchain Event**: DoctorRespondsToReview
- **Note**: Prescription goes back to pharmacist for re-review

### Action: PHARMACIST_APPROVE
- **Pharmacy Status**: PENDING_REVIEW/UNDER_REVIEW → APPROVED
- **Doctor Status**: pending_review/under_review → pharmacy_approved
- **Actor**: Pharmacist
- **Blockchain Event**: PharmacistDecision

### Action: PHARMACIST_DENY
- **Pharmacy Status**: PENDING_REVIEW/UNDER_REVIEW → DENIED
- **Doctor Status**: pending_review/under_review → pharmacy_denied
- **Actor**: Pharmacist
- **Blockchain Event**: PharmacistDecision

### Action: DOCTOR_CANCEL
- **Pharmacy Status**: UNDER_REVIEW → CANCELLED
- **Doctor Status**: under_review → cancelled
- **Actor**: Doctor
- **Blockchain Event**: DoctorCancelsPrescription

## Status Update Endpoints

### When Pharmacy Updates Status → Notify Doctor
Pharmacy app calls: `POST http://localhost:8003/api/prescription-status-update`

```json
{
  "prescription_uuid": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pharmacy_approved" | "pharmacy_denied" | "under_review" | "cancelled",
  "pharmacy_notes": "Optional explanation"
}
```

### When Doctor Updates Status → Notify Pharmacy
Doctor app calls: `POST http://localhost:8004/api/prescription-status-update`

```json
{
  "prescription_uuid": "550e8400-e29b-41d4-a716-446655440000",
  "status": "UNDER_REVIEW" | "CANCELLED",
  "doctor_notes": "Optional explanation"
}
```

## Database Synchronization

Both systems maintain a `prescription_communications` table that logs all status changes:

```sql
INSERT INTO prescription_communications (
    prescription_uuid,
    action_type,
    actor_type,
    actor_id,
    actor_name,
    comments,
    previous_status,
    new_status,
    created_at
) VALUES (...);
```

This ensures:
1. Complete audit trail
2. Ability to reconstruct conversation history
3. Synchronization point between systems
4. Blockchain verification data

## Example: Full Cycle Workflow

1. **Doctor creates prescription**
   - Doctor DB: `draft`
   - Pharmacy DB: N/A

2. **Doctor sends to pharmacy**
   - Doctor DB: `draft` → `sent_to_pharmacy`
   - Pharmacy DB: N/A → `RECEIVED`
   - Blockchain: PrescriptionCreated

3. **AI processes**
   - Doctor DB: `sent_to_pharmacy` (no change)
   - Pharmacy DB: `RECEIVED` → `AI_FLAGGED`
   - Blockchain: AIReviewCompleted

4. **Pharmacist requests review**
   - Doctor DB: `sent_to_pharmacy` → `under_review`
   - Pharmacy DB: `AI_FLAGGED` → `UNDER_REVIEW`
   - Blockchain: PharmacistRequestsReview
   - Communication logged in both DBs

5. **Doctor responds with explanation**
   - Doctor DB: `under_review` (no change)
   - Pharmacy DB: `UNDER_REVIEW` (no change)
   - Blockchain: DoctorRespondsToReview
   - Communication logged in both DBs

6. **Pharmacist approves**
   - Doctor DB: `under_review` → `pharmacy_approved`
   - Pharmacy DB: `UNDER_REVIEW` → `APPROVED`
   - Blockchain: PharmacistDecision
   - Communication logged in both DBs

## Implementation Notes

1. **Always update both databases** when status changes
2. **Always log to blockchain** for immutable audit trail
3. **Always create communication record** for conversation history
4. **Use async HTTP calls** to avoid blocking
5. **Handle failures gracefully** - log errors but don't block workflow
6. **Maintain idempotency** - duplicate events should be safe

## Testing Status Transitions

```python
# Test script to verify all status transitions
test_cases = [
    # (initial_status, action, expected_status)
    ("AI_FLAGGED", "PHARMACIST_REQUEST_REVIEW", "UNDER_REVIEW"),
    ("UNDER_REVIEW", "DOCTOR_RESPONSE", "UNDER_REVIEW"),
    ("UNDER_REVIEW", "PHARMACIST_APPROVE", "APPROVED"),
    ("UNDER_REVIEW", "PHARMACIST_DENY", "DENIED"),
    ("UNDER_REVIEW", "DOCTOR_CANCEL", "CANCELLED"),
]
```
