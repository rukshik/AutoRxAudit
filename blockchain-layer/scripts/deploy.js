// Deploy PrescriptionAuditContract to local Hardhat network
const hre = require("hardhat");

async function main() {
  console.log("Deploying PrescriptionAuditContract...");

  // Get deployer account
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying with account:", deployer.address);

  // Deploy contract
  const PrescriptionAuditContract = await hre.ethers.getContractFactory("PrescriptionAuditContract");
  const contract = await PrescriptionAuditContract.deploy();
  await contract.waitForDeployment();

  const contractAddress = await contract.getAddress();
  console.log("✓ PrescriptionAuditContract deployed to:", contractAddress);

  // Save deployment info to .env format
  console.log("\n===========================================");
  console.log("Add this to blockchain-layer/.env file:");
  console.log("===========================================");
  console.log(`BLOCKCHAIN_RPC_URL=http://127.0.0.1:8545`);
  console.log(`CONTRACT_ADDRESS=${contractAddress}`);
  console.log(`DEPLOYER_PRIVATE_KEY=${deployer.privateKey}`);
  console.log("===========================================\n");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
