// Deploy PharmacyWorkflowAudit contract to local Hardhat network
const hre = require("hardhat");

async function main() {
  console.log("Deploying PharmacyWorkflowAudit contract...");

  // Get deployer account
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying with account:", deployer.address);
  console.log("Deployer private key:", deployer.privateKey);

  // Deploy contract
  const PharmacyWorkflowAudit = await hre.ethers.getContractFactory("PharmacyWorkflowAudit");
  const contract = await PharmacyWorkflowAudit.deploy();
  await contract.waitForDeployment();

  const contractAddress = await contract.getAddress();
  console.log("✓ PharmacyWorkflowAudit deployed to:", contractAddress);
  console.log("PHARMACY_WORKFLOW_CONTRACT_ADDRESS:", contractAddress);
  console.log("Deployer private key:", deployer.privateKey);

  // Save deployment info
  console.log("\n===========================================");
  console.log("Hardhat Runtime Environment (hre):", hre);
  console.log("Add this to blockchain-layer/.env file:");
  console.log("===========================================");
  console.log(`PHARMACY_WORKFLOW_CONTRACT_ADDRESS=${contractAddress}`);
  console.log("===========================================\n");
  
  // Get contract statistics
  const stats = await contract.getStatistics();
  console.log("Contract initialized with:");
  console.log(`  Total Prescriptions: ${stats[0]}`);
  console.log(`  Total AI Reviews: ${stats[1]}`);
  console.log(`  Total Pharmacist Decisions: ${stats[2]}`);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
