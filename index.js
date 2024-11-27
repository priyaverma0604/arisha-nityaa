const fetchTransactions = require("./fetchTransactions");
const { formatTransactionData } = require("./formatData");

const main = async () => {
  try {
    console.log("Fetching Ethereum Transactions...");
    const transactions = await fetchTransactions(5); // Fetch the last 5 blocks
    const formattedData = formatTransactionData(transactions);

    console.log("Formatted Transaction Data:");
    console.log(formattedData);
  } catch (error) {
    console.error("Error:", error);
  }
};

main();
