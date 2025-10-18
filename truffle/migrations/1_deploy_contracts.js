const PrescriptionAuditContract = artifacts.require("PrescriptionAuditContract");

module.exports = function (deployer) {
  deployer.deploy(PrescriptionAuditContract);
};