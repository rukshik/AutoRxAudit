# Enhanced Doctor-Pharmacist Collaborative Review Workflow

## Overview
This document outlines the implementation of a collaborative review workflow between doctors and pharmacists, enabling back-and-forth communication for prescription reviews with full blockchain audit trail.

## Workflow States

### Status Values:
- **RECEIVED** - Prescription received, awaiting AI processing
- **AI_APPROVED** - AI approved without flags
- **AI_FLAGGED** - AI flagged for review
- **AI_ERROR** - AI processing error
- **PENDING_REVIEW** - Awaiting pharmacist review (after AI completion)
- **UNDER_REVIEW** - Pharmacist requested more information from doctor
- **APPROVED** - Pharmacist approved
- **DENIED** - Pharmacist denied
- **CANCELLED** - Doctor cancelled

## Workflow Flow

```
Doctor Creates Prescription
         ↓
    [AI Processing]
         ↓
   PENDING_REVIEW
         ↓
    Pharmacist Reviews
         ↓
   ┌─────┴─────┬─────────────┐
   ↓           ↓             ↓
APPROVE     DENY      REQUEST REVIEW
   ↓           ↓             ↓
[APPROVED]  [DENIED]   [UNDER_REVIEW]
                             ↓
                       Doctor Reviews
                             ↓
                    ┌────────┴────────┐
                    ↓                 ↓
              ADD EXPLANATION    CANCEL
                    ↓                 ↓
            [UNDER_REVIEW]      [CANCELLED]
             (back to            
          Pharmacist Review)
```

## Database Schema

### Table: `prescription_communications`

| Column | Type | Description |
|--------|------|-------------|
| communication_id | SERIAL | Primary key |
| prescription_uuid | VARCHAR(100) | Prescription identifier |
| prescription_id | INTEGER | FK to prescription_requests |
| action_type | VARCHAR(50) | Action performed |
| actor_type | VARCHAR(20) | PHARMACIST or DOCTOR |
| actor_id | INTEGER | User ID |
| actor_name | VARCHAR(200) | User name |
| comments | TEXT | Message/explanation |
| previous_status | VARCHAR(50) | Status before action |
| new_status | VARCHAR(50) | Status after action |
| created_at | TIMESTAMP | When action occurred |

### Action Types:
- **PHARMACIST_REQUEST_REVIEW** - Pharmacist asks for more info
- **DOCTOR_RESPONSE** - Doctor provides explanation
- **PHARMACIST_APPROVE** - Pharmacist approves
- **PHARMACIST_DENY** - Pharmacist denies
- **DOCTOR_CANCEL** - Doctor cancels prescription

## Blockchain Events

### New Events in PharmacyWorkflowAudit Contract:

1. **PharmacistRequestsReview**
   - prescriptionUuid
   - pharmacistId
   - pharmacistName
   - reviewComments
   - timestamp

2. **DoctorRespondsToReview**
   - prescriptionUuid
   - doctorId
   - doctorName
   - responseComments
   - timestamp

3. **DoctorCancelsPrescription**
   - prescriptionUuid
   - doctorId
   - cancellationReason
   - timestamp

## API Endpoints

### Pharmacy App (New/Updated):

**POST /api/prescription-action** (Updated)
```json
{
  "prescription_id": 123,
  "action": "APPROVE" | "DENY" | "REQUEST_REVIEW",
  "comments": "Optional explanation"
}
```

### Doctor App (New):

**GET /api/prescriptions-under-review**
- Returns prescriptions in UNDER_REVIEW status with pharmacist comments

**POST /api/prescription-respond**
```json
{
  "prescription_uuid": "550e8400-...",
  "action": "PROVIDE_EXPLANATION" | "CANCEL",
  "comments": "Required explanation"
}
```

## Implementation Steps

### Phase 1: Database Setup ✅
- [x] Create `prescription_communications` table (pharmacy DB)
- [x] Create `prescription_communications` table (doctor DB)
- [x] Add new status constraints
- [x] Create indexes

### Phase 2: Smart Contract Updates ✅
- [x] Add new events (PharmacistRequestsReview, DoctorRespondsToReview, DoctorCancelsPrescription)
- [x] Add counter variables
- [x] Create logging functions
- [ ] Redeploy contract
- [ ] Update ABI in blockchain service

### Phase 3: Pharmacy App Updates
- [ ] Update prescription action endpoint to support REQUEST_REVIEW
- [ ] Add function to log communications to database
- [ ] Add blockchain logging for new actions
- [ ] Update UI to show communication history
- [ ] Add REQUEST_REVIEW button with comment modal

### Phase 4: Doctor App Updates
- [ ] Create endpoint to fetch prescriptions under review
- [ ] Create endpoint to respond to review requests
- [ ] Add function to log communications to database
- [ ] Add blockchain logging for doctor responses
- [ ] Update UI to show review requests
- [ ] Add response modal with explanation field
- [ ] Add cancel option with reason field

### Phase 5: Blockchain Service Updates
- [ ] Add endpoint: POST /pharmacy/request-review
- [ ] Add endpoint: POST /doctor/respond-to-review
- [ ] Add endpoint: POST /doctor/cancel-prescription
- [ ] Update event listeners for new events

### Phase 6: UI Enhancements
- [ ] Pharmacy: Add "Request Review" button
- [ ] Pharmacy: Show communication history timeline
- [ ] Doctor: Add "Under Review" section
- [ ] Doctor: Show AI scores and pharmacist comments
- [ ] Doctor: Add response form
- [ ] Both: Visual indicators for review cycles

### Phase 7: Testing
- [ ] Test full workflow: Approve path
- [ ] Test full workflow: Deny path
- [ ] Test full workflow: Review cycle (1 round)
- [ ] Test full workflow: Review cycle (multiple rounds)
- [ ] Test full workflow: Cancel path
- [ ] Verify blockchain logging for all actions
- [ ] Verify database synchronization

## Example Workflow Scenario

### Scenario: Pharmacist Requests Review

1. **Doctor creates prescription** (Oxycodone 5mg)
   - Status: RECEIVED → AI_FLAGGED
   - Blockchain: PrescriptionCreated + AIReviewCompleted

2. **Pharmacist reviews** and sees:
   - Eligibility: 45% (Not Eligible)
   - OUD Risk: 75% (High Risk)
   - Flag Reason: "No clinical need for opioids AND High OUD risk detected"

3. **Pharmacist clicks "Request Review"**
   - Comments: "Patient has no documented pain diagnosis. Please provide clinical justification for opioid prescription."
   - Status: AI_FLAGGED → UNDER_REVIEW
   - Blockchain: PharmacistRequestsReview logged
   - Database: Communication record created
   - Notification sent to doctor

4. **Doctor sees review request** in their dashboard
   - Views AI scores
   - Views pharmacist comments
   - Options: Provide Explanation | Cancel

5. **Doctor provides explanation**
   - Comments: "Patient has chronic lower back pain from work injury (2023-05-15). Conservative treatments (PT, NSAIDs) failed. MRI shows disc herniation L4-L5."
   - Status: UNDER_REVIEW (stays)
   - Blockchain: DoctorRespondsToReview logged
   - Database: Communication record created
   - Notification sent to pharmacist

6. **Pharmacist reviews doctor's explanation**
   - Reads justification
   - Options: Approve | Deny | Request More Review

7. **Pharmacist approves**
   - Comments: "Thank you for clarification. Clinical justification provided."
   - Status: UNDER_REVIEW → APPROVED
   - Blockchain: PharmacistDecision logged
   - Database: Communication record created
   - Both systems updated

## Benefits

1. **Improved Clinical Decision Making**
   - Collaborative review ensures best patient outcomes
   - AI + Human expertise combination

2. **Complete Audit Trail**
   - Every communication logged to blockchain
   - Full transparency for regulatory compliance
   - Immutable record of review process

3. **Reduced Errors**
   - Catch inappropriate prescriptions before dispensing
   - Allow doctors to provide context AI might miss

4. **Better Doctor-Pharmacist Relationship**
   - Professional dialogue
   - Mutual respect for expertise
   - Shared responsibility

5. **Regulatory Compliance**
   - DEA requirements for controlled substances
   - Documentation of due diligence
   - Evidence of systematic review

## Security Considerations

- All communications encrypted in transit
- Access control: Only authorized users can view/respond
- Rate limiting on review requests to prevent abuse
- Audit logging of all database access
- Blockchain provides tamper-proof record

## Performance Considerations

- Index on prescription_uuid for fast lookups
- Pagination for communication history
- Async blockchain logging (non-blocking)
- Cache frequently accessed prescription data

## Future Enhancements

- Email/SMS notifications for review requests
- Real-time updates using WebSockets
- Mobile app support
- Integration with EHR systems
- Analytics dashboard for review patterns
- Machine learning to predict review outcomes
