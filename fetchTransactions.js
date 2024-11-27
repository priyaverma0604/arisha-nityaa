const { ethers } = require("ethers");
const provider = require("./provider"); // Adjust path if necessary

const fetchTransactions = async (blockCount = 1) => {
  try {
    const latestBlockNumber = await provider.getBlockNumber();
    const transactions = [];

    for (let i = 0; i < blockCount; i++) {
      const blockNumber = latestBlockNumber - i;
      console.log(`Fetching Block ${blockNumber}...`);

      const block = await provider.getBlock(blockNumber);
      if (block && block.transactions.length > 0) {
        console.log(
          `Block ${blockNumber} has ${block.transactions.length} transactions`
        );

        for (const txHash of block.transactions) {
          const tx = await provider.getTransaction(txHash);

          if (tx && tx.from && tx.to && tx.value) {
            const etherValue = ethers.formatEther(tx.value);

            transactions.push({
              from: tx.from,
              to: tx.to,
              value: etherValue,
              hash: tx.hash,
            });

            console.log(`Transaction: ${tx.hash}`);
            console.log(`From: ${tx.from}`);
            console.log(`To: ${tx.to}`);
            console.log(`Value: ${etherValue} ETH`);
          } else {
            console.log(
              `Transaction ${txHash} does not have complete details.`
            );
          }
        }
      } else {
        console.log(`No transactions found in block ${blockNumber}`);
      }
    }

    console.log("Filtered Transactions:", transactions); // Valid transactions
    return transactions;
  } catch (error) {
    console.error("Error fetching transactions:", error);
    throw error;
  }
};

module.exports = fetchTransactions;
