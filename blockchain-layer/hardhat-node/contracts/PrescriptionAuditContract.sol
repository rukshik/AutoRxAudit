// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title PrescriptionAuditContract
 * @dev Smart contract for tracking flagged prescriptions and override actions
 * @author AutoRxAudit System
 */
contract PrescriptionAuditContract {
    
    // Input struct to avoid stack too deep error in recordAudit function
    struct AuditInput {
        uint256 auditId;
        uint256 prescriptionId;
        string patientId;
        string drugName;
        uint8 eligibilityScore;
        uint8 eligibilityPrediction;
        uint8 oudRiskScore;
        uint8 oudRiskPrediction;
        bool flagged;
        string flagReason;
        string recommendation;
    }
    
    // Struct to represent a prescription audit record (matches audit_logs table)
    struct PrescriptionRecord {
        uint256 blockchainId;        // Blockchain-specific ID
        uint256 auditId;             // audit_id from database
        uint256 prescriptionId;      // prescription_id from database
        string patientId;            // patient_id
        string drugName;             // drug_name
        uint8 eligibilityScore;      // eligibility_score (0-100)
        uint8 eligibilityPrediction; // eligibility_prediction (0 or 1)
        uint8 oudRiskScore;          // oud_risk_score (0-100)
        uint8 oudRiskPrediction;     // oud_risk_prediction (0 or 1)
        bool flagged;                // flagged
        string flagReason;           // flag_reason
        string recommendation;       // recommendation
        uint256 auditedAt;           // audited_at timestamp
        string reviewedBy;           // reviewed_by (user_id as string)
        string reviewedByName;       // reviewer's full name
        string reviewedByEmail;      // reviewer's email
        string action;               // action (APPROVED, DENIED, etc.)
        string actionReason;         // action_reason
        uint256 reviewedAt;          // reviewed_at timestamp
    }
    
    // Events for tracking actions (matches database workflow)
    event AuditRecorded(
        uint256 indexed blockchainId,
        uint256 indexed auditId,
        uint256 prescriptionId,
        string patientId,
        bool flagged,
        uint8 oudRiskScore,
        uint256 timestamp
    );
    
    event PharmacistActionRecorded(
        uint256 indexed blockchainId,
        uint256 indexed auditId,
        string action,
        string reviewedBy,
        uint256 timestamp
    );
    
    // State variables
    address public owner;
    mapping(uint256 => PrescriptionRecord) public auditRecords; // blockchainId => record
    mapping(uint256 => uint256) public auditIdToBlockchainId;   // auditId => blockchainId
    mapping(string => uint256[]) public patientAudits;          // patientId => blockchainIds
    
    uint256 public recordCounter = 0;
    
    // Access control
    constructor() {
        owner = msg.sender;
    }
    
    /**
     * @dev Record audit result from main API (after AI model evaluation)
     * @param input Struct containing all audit data
     */
    function recordAudit(AuditInput memory input) external returns (uint256) {
        recordCounter++;
        uint256 blockchainId = recordCounter;
        
        PrescriptionRecord memory newRecord = PrescriptionRecord({
            blockchainId: blockchainId,
            auditId: input.auditId,
            prescriptionId: input.prescriptionId,
            patientId: input.patientId,
            drugName: input.drugName,
            eligibilityScore: input.eligibilityScore,
            eligibilityPrediction: input.eligibilityPrediction,
            oudRiskScore: input.oudRiskScore,
            oudRiskPrediction: input.oudRiskPrediction,
            flagged: input.flagged,
            flagReason: input.flagReason,
            recommendation: input.recommendation,
            auditedAt: block.timestamp,
            reviewedBy: "",
            reviewedByName: "",
            reviewedByEmail: "",
            action: "",
            actionReason: "",
            reviewedAt: 0
        });
        
        auditRecords[blockchainId] = newRecord;
        auditIdToBlockchainId[input.auditId] = blockchainId;
        patientAudits[input.patientId].push(blockchainId);
        
        emit AuditRecorded(
            blockchainId,
            input.auditId,
            input.prescriptionId,
            input.patientId,
            input.flagged,
            input.oudRiskScore,
            block.timestamp
        );
        
        return blockchainId;
    }
    
    /**
     * @dev Record pharmacist action (approve/deny/override)
     * Creates a new immutable record for each action to maintain full audit trail
     * @param auditId audit_id from database
     * @param action action (APPROVED, DENIED, OVERRIDE_APPROVE, OVERRIDE_DENY)
     * @param actionReason action_reason
     * @param reviewedBy reviewed_by (user_id as string)
     * @param reviewedByName reviewer's full name
     * @param reviewedByEmail reviewer's email
     */
    function recordPharmacistAction(
        uint256 auditId,
        string memory action,
        string memory actionReason,
        string memory reviewedBy,
        string memory reviewedByName,
        string memory reviewedByEmail
    ) external returns (uint256) {
        uint256 originalBlockchainId = auditIdToBlockchainId[auditId];
        require(originalBlockchainId != 0, "Audit record not found");
        
        // Get original record to copy data
        PrescriptionRecord memory originalRecord = auditRecords[originalBlockchainId];
        
        // Create new record with updated action
        recordCounter++;
        uint256 newBlockchainId = recordCounter;
        
        PrescriptionRecord memory newRecord = PrescriptionRecord({
            blockchainId: newBlockchainId,
            auditId: auditId,
            prescriptionId: originalRecord.prescriptionId,
            patientId: originalRecord.patientId,
            drugName: originalRecord.drugName,
            eligibilityScore: originalRecord.eligibilityScore,
            eligibilityPrediction: originalRecord.eligibilityPrediction,
            oudRiskScore: originalRecord.oudRiskScore,
            oudRiskPrediction: originalRecord.oudRiskPrediction,
            flagged: originalRecord.flagged,
            flagReason: originalRecord.flagReason,
            recommendation: originalRecord.recommendation,
            auditedAt: originalRecord.auditedAt,
            reviewedBy: reviewedBy,
            reviewedByName: reviewedByName,
            reviewedByEmail: reviewedByEmail,
            action: action,
            actionReason: actionReason,
            reviewedAt: block.timestamp
        });
        
        auditRecords[newBlockchainId] = newRecord;
        patientAudits[originalRecord.patientId].push(newBlockchainId);
        
        emit PharmacistActionRecorded(
            newBlockchainId,
            auditId,
            action,
            reviewedBy,
            block.timestamp
        );
        
        return newBlockchainId;
    }
    
    /**
     * @dev Get audit record by blockchain ID
     * @param blockchainId Blockchain-specific ID
     */
    function getAuditRecord(uint256 blockchainId) external view returns (PrescriptionRecord memory) {
        return auditRecords[blockchainId];
    }
    
    /**
     * @dev Get blockchain ID by audit ID
     * @param auditId audit_id from database
     */
    function getBlockchainIdByAuditId(uint256 auditId) external view returns (uint256) {
        return auditIdToBlockchainId[auditId];
    }
    
    /**
     * @dev Get all audit records for a patient
     * @param patientId Patient identifier
     */
    function getPatientAudits(string memory patientId) external view returns (uint256[] memory) {
        return patientAudits[patientId];
    }
    
    /**
     * @dev Get total audit record count
     */
    function getTotalRecords() external view returns (uint256) {
        return recordCounter;
    }
}
