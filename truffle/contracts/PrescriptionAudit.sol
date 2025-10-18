// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title PrescriptionAuditContract
 * @dev Smart contract for tracking flagged prescriptions and override actions
 * @author AutoRxAudit System
 */
contract PrescriptionAuditContract {
    
    // Struct to represent a prescription record
    struct PrescriptionRecord {
        uint256 prescriptionId;
        string patientId;
        string doctorId;
        string pharmacyId;
        string drugName;
        uint256 quantity;
        uint256 timestamp;
        uint8 riskScore; // 0-100 risk score from AI
        string riskFactors; // JSON string of risk factors
        bool isFlagged;
        bool isOverridden;
        string overrideReason;
        string overrideBy; // Doctor/pharmacist who overrode
        uint256 overrideTimestamp;
    }
    
    // Events for tracking actions
    event PrescriptionFlagged(
        uint256 indexed prescriptionId,
        string patientId,
        string doctorId,
        string pharmacyId,
        uint8 riskScore,
        string riskFactors
    );
    
    event PrescriptionOverridden(
        uint256 indexed prescriptionId,
        string overrideBy,
        string overrideReason,
        uint256 timestamp
    );
    
    event PrescriptionVerified(
        uint256 indexed prescriptionId,
        string verifiedBy,
        uint256 timestamp
    );
    
    // State variables
    address public owner;
    mapping(uint256 => PrescriptionRecord) public prescriptions;
    mapping(string => uint256[]) public patientPrescriptions; // patientId => prescriptionIds
    mapping(string => uint256[]) public doctorPrescriptions; // doctorId => prescriptionIds
    mapping(string => uint256[]) public pharmacyPrescriptions; // pharmacyId => prescriptionIds
    
    uint256 public prescriptionCounter = 0;
    uint256 public constant RISK_THRESHOLD = 70; // Flag prescriptions with risk score >= 70
    
    // Access control
    mapping(address => bool) public authorizedDoctors;
    mapping(address => bool) public authorizedPharmacists;
    mapping(address => bool) public authorizedAuditors;
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    modifier onlyAuthorized() {
        require(
            authorizedDoctors[msg.sender] || 
            authorizedPharmacists[msg.sender] || 
            authorizedAuditors[msg.sender] || 
            msg.sender == owner,
            "Not authorized"
        );
        _;
    }
    
    constructor() {
        owner = msg.sender;
    }
    
    /**
     * @dev Add authorized doctor
     */
    function addAuthorizedDoctor(address doctor) external onlyOwner {
        authorizedDoctors[doctor] = true;
    }
    
    /**
     * @dev Add authorized pharmacist
     */
    function addAuthorizedPharmacist(address pharmacist) external onlyOwner {
        authorizedPharmacists[pharmacist] = true;
    }
    
    /**
     * @dev Add authorized auditor
     */
    function addAuthorizedAuditor(address auditor) external onlyOwner {
        authorizedAuditors[auditor] = true;
    }
    
    /**
     * @dev Record a flagged prescription
     * @param patientId Patient identifier
     * @param doctorId Doctor identifier
     * @param pharmacyId Pharmacy identifier
     * @param drugName Name of the prescribed drug
     * @param quantity Quantity prescribed
     * @param riskScore AI-calculated risk score (0-100)
     * @param riskFactors JSON string of risk factors
     */
    function recordFlaggedPrescription(
        string memory patientId,
        string memory doctorId,
        string memory pharmacyId,
        string memory drugName,
        uint256 quantity,
        uint8 riskScore,
        string memory riskFactors
    ) external onlyAuthorized returns (uint256) {
        require(riskScore >= RISK_THRESHOLD, "Risk score below threshold");
        
        prescriptionCounter++;
        uint256 prescriptionId = prescriptionCounter;
        
        PrescriptionRecord memory newPrescription = PrescriptionRecord({
            prescriptionId: prescriptionId,
            patientId: patientId,
            doctorId: doctorId,
            pharmacyId: pharmacyId,
            drugName: drugName,
            quantity: quantity,
            timestamp: block.timestamp,
            riskScore: riskScore,
            riskFactors: riskFactors,
            isFlagged: true,
            isOverridden: false,
            overrideReason: "",
            overrideBy: "",
            overrideTimestamp: 0
        });
        
        prescriptions[prescriptionId] = newPrescription;
        patientPrescriptions[patientId].push(prescriptionId);
        doctorPrescriptions[doctorId].push(prescriptionId);
        pharmacyPrescriptions[pharmacyId].push(prescriptionId);
        
        emit PrescriptionFlagged(
            prescriptionId,
            patientId,
            doctorId,
            pharmacyId,
            riskScore,
            riskFactors
        );
        
        return prescriptionId;
    }
    
    /**
     * @dev Override a flagged prescription
     * @param prescriptionId ID of the prescription to override
     * @param overrideReason Reason for override
     * @param overrideBy Identifier of person overriding
     */
    function overridePrescription(
        uint256 prescriptionId,
        string memory overrideReason,
        string memory overrideBy
    ) external onlyAuthorized {
        require(prescriptions[prescriptionId].prescriptionId != 0, "Prescription not found");
        require(prescriptions[prescriptionId].isFlagged, "Prescription not flagged");
        require(!prescriptions[prescriptionId].isOverridden, "Already overridden");
        
        prescriptions[prescriptionId].isOverridden = true;
        prescriptions[prescriptionId].overrideReason = overrideReason;
        prescriptions[prescriptionId].overrideBy = overrideBy;
        prescriptions[prescriptionId].overrideTimestamp = block.timestamp;
        
        emit PrescriptionOverridden(
            prescriptionId,
            overrideBy,
            overrideReason,
            block.timestamp
        );
    }
    
    /**
     * @dev Verify a prescription (for pharmacy use)
     * @param prescriptionId ID of the prescription to verify
     * @param verifiedBy Identifier of person verifying
     */
    function verifyPrescription(
        uint256 prescriptionId,
        string memory verifiedBy
    ) external onlyAuthorized {
        require(prescriptions[prescriptionId].prescriptionId != 0, "Prescription not found");
        
        emit PrescriptionVerified(
            prescriptionId,
            verifiedBy,
            block.timestamp
        );
    }
    
    /**
     * @dev Get prescription details
     * @param prescriptionId ID of the prescription
     */
    function getPrescription(uint256 prescriptionId) external view returns (
        uint256 id,
        string memory patientId,
        string memory doctorId,
        string memory pharmacyId,
        string memory drugName,
        uint256 quantity,
        uint256 timestamp,
        uint8 riskScore,
        string memory riskFactors,
        bool isFlagged,
        bool isOverridden,
        string memory overrideReason,
        string memory overrideBy,
        uint256 overrideTimestamp
    ) {
        PrescriptionRecord memory prescription = prescriptions[prescriptionId];
        return (
            prescription.prescriptionId,
            prescription.patientId,
            prescription.doctorId,
            prescription.pharmacyId,
            prescription.drugName,
            prescription.quantity,
            prescription.timestamp,
            prescription.riskScore,
            prescription.riskFactors,
            prescription.isFlagged,
            prescription.isOverridden,
            prescription.overrideReason,
            prescription.overrideBy,
            prescription.overrideTimestamp
        );
    }
    
    /**
     * @dev Get all prescriptions for a patient
     * @param patientId Patient identifier
     */
    function getPatientPrescriptions(string memory patientId) external view returns (uint256[] memory) {
        return patientPrescriptions[patientId];
    }
    
    /**
     * @dev Get all prescriptions for a doctor
     * @param doctorId Doctor identifier
     */
    function getDoctorPrescriptions(string memory doctorId) external view returns (uint256[] memory) {
        return doctorPrescriptions[doctorId];
    }
    
    /**
     * @dev Get all prescriptions for a pharmacy
     * @param pharmacyId Pharmacy identifier
     */
    function getPharmacyPrescriptions(string memory pharmacyId) external view returns (uint256[] memory) {
        return pharmacyPrescriptions[pharmacyId];
    }
    
    /**
     * @dev Get total prescription count
     */
    function getTotalPrescriptions() external view returns (uint256) {
        return prescriptionCounter;
    }
    
    /**
     * @dev Check if prescription is flagged
     * @param prescriptionId ID of the prescription
     */
    function isPrescriptionFlagged(uint256 prescriptionId) external view returns (bool) {
        return prescriptions[prescriptionId].isFlagged && !prescriptions[prescriptionId].isOverridden;
    }
}