// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title PharmacyWorkflowAudit
 * @dev Smart contract for tracking prescription lifecycle from doctor to pharmacy
 * Records: prescription creation, AI review, and pharmacist decisions
 * NO patient data or drug names stored (privacy compliance)
 */
contract PharmacyWorkflowAudit {
    
    // Event 1: Prescription created by doctor
    event PrescriptionCreated(
        string indexed prescriptionUuid,
        string doctorId,
        uint256 timestamp
    );
    
    // Event 2: AI review completed
    event AIReviewCompleted(
        string indexed prescriptionUuid,
        bool flagged,
        uint8 eligibilityScore,      // 0-100
        uint8 oudRiskScore,          // 0-100
        string flagReason,
        string recommendation,
        uint256 timestamp
    );
    
    // Event 3: Pharmacist decision
    event PharmacistDecision(
        string indexed prescriptionUuid,
        string pharmacistId,
        string action,               // "APPROVED" or "DECLINED"
        string actionReason,
        uint256 timestamp
    );
    
    // State variables
    address public owner;
    uint256 public totalPrescriptions;
    uint256 public totalAIReviews;
    uint256 public totalPharmacistDecisions;
    
    // Track which prescriptions have been logged (prevents duplicate creation)
    mapping(string => bool) public prescriptionExists;
    
    constructor() {
        owner = msg.sender;
        totalPrescriptions = 0;
        totalAIReviews = 0;
        totalPharmacistDecisions = 0;
    }
    
    /**
     * @dev Log prescription creation by doctor
     * @param prescriptionUuid Unique prescription identifier
     * @param doctorId Doctor's user ID
     */
    function logPrescriptionCreated(
        string memory prescriptionUuid,
        string memory doctorId
    ) external {
        require(bytes(prescriptionUuid).length > 0, "Prescription UUID cannot be empty");
        require(bytes(doctorId).length > 0, "Doctor ID cannot be empty");
        require(!prescriptionExists[prescriptionUuid], "Prescription already logged");
        
        prescriptionExists[prescriptionUuid] = true;
        totalPrescriptions++;
        
        emit PrescriptionCreated(
            prescriptionUuid,
            doctorId,
            block.timestamp
        );
    }
    
    /**
     * @dev Log AI review completion
     * @param prescriptionUuid Unique prescription identifier
     * @param flagged Whether AI flagged for review
     * @param eligibilityScore Eligibility score (0-100)
     * @param oudRiskScore OUD risk score (0-100)
     * @param flagReason Reason for flagging (empty if not flagged)
     * @param recommendation AI recommendation
     */
    function logAIReview(
        string memory prescriptionUuid,
        bool flagged,
        uint8 eligibilityScore,
        uint8 oudRiskScore,
        string memory flagReason,
        string memory recommendation
    ) external {
        require(bytes(prescriptionUuid).length > 0, "Prescription UUID cannot be empty");
        require(eligibilityScore <= 100, "Eligibility score must be 0-100");
        require(oudRiskScore <= 100, "OUD risk score must be 0-100");
        
        totalAIReviews++;
        
        emit AIReviewCompleted(
            prescriptionUuid,
            flagged,
            eligibilityScore,
            oudRiskScore,
            flagReason,
            recommendation,
            block.timestamp
        );
    }
    
    /**
     * @dev Log pharmacist decision
     * @param prescriptionUuid Unique prescription identifier
     * @param pharmacistId Pharmacist's user ID
     * @param action "APPROVED" or "DECLINED"
     * @param actionReason Reason for decision
     */
    function logPharmacistDecision(
        string memory prescriptionUuid,
        string memory pharmacistId,
        string memory action,
        string memory actionReason
    ) external {
        require(bytes(prescriptionUuid).length > 0, "Prescription UUID cannot be empty");
        require(bytes(pharmacistId).length > 0, "Pharmacist ID cannot be empty");
        require(bytes(action).length > 0, "Action cannot be empty");
        
        totalPharmacistDecisions++;
        
        emit PharmacistDecision(
            prescriptionUuid,
            pharmacistId,
            action,
            actionReason,
            block.timestamp
        );
    }
    
    /**
     * @dev Get contract statistics
     */
    function getStatistics() external view returns (
        uint256 prescriptions,
        uint256 aiReviews,
        uint256 pharmacistDecisions
    ) {
        return (totalPrescriptions, totalAIReviews, totalPharmacistDecisions);
    }
}
