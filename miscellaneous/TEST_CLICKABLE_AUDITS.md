# Testing Clickable Pending Audits

## Changes Made

### 1. **AuditHistory.js**
- Added `onNavigateToAudit` prop
- Added `handleRowClick` function to handle pending audit clicks
- Modified table rows to be clickable when audit is pending (no action)
- Added cursor pointer and hover effect for pending rows
- Updated pending badge text to "⏳ Pending - Click to Review"

### 2. **AuditHistory.css**
- Added `.pending-clickable:hover` style for hover effect (light blue background)
- Styled `.pending-badge` for better visibility

### 3. **App.js**
- Added `selectedAudit` state to track audit selected from history
- Added `navigateToAudit(audit)` function to handle navigation with audit data
- Updated `navigateToForm` and `navigateToHistory` to clear selectedAudit
- Passed `selectedAudit` and `onAuditActionComplete` props to PrescriptionForm
- Passed `onNavigateToAudit` prop to AuditHistory

### 4. **PrescriptionForm.js**
- Added `selectedAudit` and `onAuditActionComplete` props
- Added useEffect to populate form when selectedAudit is provided
- Modified UI to show "Review Pending Audit" header with back button when viewing from history
- Hide prescription form when viewing from history (only show audit result section)
- Updated `handleAction` to navigate back to history after action is recorded
- Added 500ms delay before navigation for better UX (alert message visible)

### 5. **PrescriptionForm.css**
- Added `.audit-review-header` styles for header layout
- Added `.back-button` styles for back navigation button

## User Workflow

1. **View Audit History**: User navigates to audit history
2. **Click Pending Audit**: User clicks on a row with "⏳ Pending - Click to Review" status
3. **Review Details**: Form shows audit details with:
   - "Review Pending Audit" header
   - Back button to return to history
   - Audit result section with scores and flag reason
   - Action buttons (APPROVE, DENY, OVERRIDE_APPROVE, OVERRIDE_DENY)
   - Action reason textarea
4. **Make Decision**: User selects an action and optionally provides reason
5. **Submit**: User clicks action button
6. **Navigate Back**: After alert confirmation, automatically returns to history (500ms delay)
7. **See Updated**: Audit history refreshes showing the reviewed audit with action and reviewer info

## Testing Steps

1. **Start Backend**:
   ```powershell
   cd AutoRxAudit/api
   python app.py
   ```

2. **Start Frontend**:
   ```powershell
   cd ../frontend
   npm start
   ```

3. **Login** as pharmacist:
   - Email: `pharmacist@autorxaudit.com`
   - Password: `pharma123`

4. **Submit New Prescription** (to create pending audit):
   - Select patient with high OUD risk (e.g., 20000199)
   - Select opioid drug (e.g., Oxycodone)
   - Submit to create flagged audit

5. **Go to Audit History**:
   - Click "View Audit History" button
   - See the pending audit with "⏳ Pending - Click to Review" badge

6. **Click Pending Row**:
   - Hover over row (should show light blue background)
   - Click the row
   - Should navigate to form view with audit details

7. **Review and Take Action**:
   - Verify "Review Pending Audit" header appears
   - Verify "← Back to History" button appears
   - Verify audit result section shows scores and flag reason
   - Verify form inputs are hidden
   - Select an action (e.g., DENY)
   - Optionally add action reason
   - Click action button

8. **Verify Navigation**:
   - Should see success alert
   - After 500ms, should navigate back to history automatically
   - History should refresh and show updated audit status

9. **Verify Updated Audit**:
   - Row should no longer be clickable
   - Status should show action taken (e.g., "✓ DENIED")
   - Clinician name should show pharmacist name
   - Reviewed date should be populated

## Expected Behavior

### Pending Audits
- ✅ Clickable rows with pointer cursor
- ✅ Hover effect (light blue background)
- ✅ Badge shows "⏳ Pending - Click to Review"

### Audit Review Page
- ✅ Shows "Review Pending Audit" header
- ✅ Shows back button to return to history
- ✅ Hides prescription form inputs
- ✅ Shows audit result details
- ✅ Shows action buttons
- ✅ Allows action reason input

### After Action
- ✅ Shows success alert
- ✅ Automatically returns to history (500ms delay)
- ✅ History refreshes with updated data
- ✅ Audit shows action taken and reviewer info

### Reviewed Audits
- ✅ Not clickable (normal cursor)
- ✅ No hover effect
- ✅ Shows action badge (not pending badge)

## Troubleshooting

### Row not clickable
- Check: Is audit pending (no action value)?
- Check: Does audit have `action` field?
- Verify: `!record.action` condition in onClick handler

### Navigation not working
- Check: Is `onNavigateToAudit` prop passed correctly?
- Verify: App.js has `navigateToAudit` function
- Check: Console for errors

### Audit details not showing
- Check: Is `selectedAudit` prop received?
- Verify: useEffect in PrescriptionForm sets auditResult
- Check: Console log selectedAudit value

### Not returning to history after action
- Check: Is `onAuditActionComplete` prop passed?
- Verify: setTimeout executes after 500ms
- Check: Alert appears before navigation

## Files Modified

1. `frontend/src/AuditHistory.js` - Added clickable rows
2. `frontend/src/AuditHistory.css` - Added hover styles
3. `frontend/src/App.js` - Added navigation logic
4. `frontend/src/PrescriptionForm.js` - Added audit review mode
5. `frontend/src/PrescriptionForm.css` - Added review header styles
