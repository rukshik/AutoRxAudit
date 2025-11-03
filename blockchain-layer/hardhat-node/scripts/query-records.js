// Script to query all blockchain records directly from the smart contract

const hre = require("hardhat");

async function main() {
  // Contract address - update this if you redeployed
  const contractAddress = "0x5FbDB2315678afecb367f032d93F642f64180aa3";

  // Get the contract
  const PrescriptionAuditContract = await hre.ethers.getContractFactory("PrescriptionAuditContract");
  const contract = PrescriptionAuditContract.attach(contractAddress);

  console.log("\n=== BLOCKCHAIN AUDIT RECORDS ===\n");

  try {
    // Get total number of records
    const totalRecords = await contract.getTotalRecords();
    console.log(`Total Records on Blockchain: ${totalRecords.toString()}\n`);

    if (totalRecords.toString() === "0") {
      console.log("No records found on blockchain.\n");
      return;
    }

    // Query each record
    for (let i = 1; i <= totalRecords; i++) {
      console.log(`\n--- Record #${i} ---`);
      
      const record = await contract.getAuditRecord(i);
      
      console.log(`Blockchain ID: ${record.blockchainId.toString()}`);
      console.log(`Audit ID: ${record.auditId.toString()}`);
      console.log(`Prescription ID: ${record.prescriptionId.toString()}`);
      console.log(`Patient ID: ${record.patientId}`);
      console.log(`Drug Name: ${record.drugName}`);
      console.log(`Eligibility Score: ${record.eligibilityScore}%`);
      console.log(`Eligibility Prediction: ${record.eligibilityPrediction === 1 ? "Eligible" : "Not Eligible"}`);
      console.log(`OUD Risk Score: ${record.oudRiskScore}%`);
      console.log(`OUD Risk Prediction: ${record.oudRiskPrediction === 1 ? "High Risk" : "Low Risk"}`);
      console.log(`Flagged: ${record.flagged ? "YES ⚠️" : "NO ✅"}`);
      console.log(`Flag Reason: ${record.flagReason}`);
      console.log(`Recommendation: ${record.recommendation}`);
      
      const auditedDate = new Date(Number(record.auditedAt) * 1000);
      console.log(`Audited At: ${auditedDate.toLocaleString()}`);
      
      if (record.reviewedBy) {
        console.log(`\n--- Review Information ---`);
        console.log(`Reviewed By: ${record.reviewedBy}`);
        console.log(`Reviewer Name: ${record.reviewedByName}`);
        console.log(`Reviewer Email: ${record.reviewedByEmail}`);
        console.log(`Action: ${record.action}`);
        console.log(`Action Reason: ${record.actionReason}`);
        
        if (Number(record.reviewedAt) > 0) {
          const reviewedDate = new Date(Number(record.reviewedAt) * 1000);
          console.log(`Reviewed At: ${reviewedDate.toLocaleString()}`);
        }
      } else {
        console.log(`\nReview Status: PENDING ⏳`);
      }
      
      console.log(`${"=".repeat(60)}`);
    }

    console.log("\n✅ Query complete!\n");

  } catch (error) {
    console.error("\n❌ Error querying blockchain:");
    console.error(error.message);
    
    if (error.message.includes("call revert exception")) {
      console.log("\n💡 Tip: Make sure the contract is deployed and the address is correct.");
      console.log("Run: npx hardhat run scripts/deploy.js --network localhost\n");
    }
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
