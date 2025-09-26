import 'dotenv/config';
import { ethers } from 'ethers';

async function main() {
  console.log('🔍 Verifying Private Key for Owner Wallet...');

  // Owner wallet address
  const ownerAddress = '0x058C8FE01E5c9eaC6ee19e6673673B549B368843';

  // Current private key from .env
  const currentPrivateKey = process.env.PRIVATE_KEY;

  if (!currentPrivateKey) {
    console.log('❌ No PRIVATE_KEY found in .env file');
    return;
  }

  // Create wallet from private key
  const wallet = new ethers.Wallet(currentPrivateKey);
  const generatedAddress = await wallet.getAddress();

  console.log('📍 Owner Address:', ownerAddress);
  console.log('🔑 Generated Address:', generatedAddress);
  const isMatch = ownerAddress.toLowerCase() === generatedAddress.toLowerCase();
  console.log('✅ Addresses match:', isMatch);

  if (isMatch) {
    console.log('🎉 Private key is correct for owner wallet!');

    // Provider from env or public fallback
    const rpcUrl =
      process.env.RPC_URL ||
      process.env.ETH_RPC_URL ||
      process.env.ALCHEMY_URL ||
      process.env.ANVIL_URL ||
      'https://ethereum.publicnode.com';

    const provider = new ethers.JsonRpcProvider(rpcUrl);

    // Check ETH balance
    const ethBalance = await provider.getBalance(ownerAddress);
    console.log('💰 ETH Balance:', ethers.formatEther(ethBalance), 'ETH');

    // Check ETHGR token balance
    const ethgrContract = new ethers.Contract(
      '0x60762856508e45eb4F011EA8F5D4D421e85eb41D',
      [
        'function balanceOf(address) view returns (uint256)',
        'function decimals() view returns (uint8)',
        'function symbol() view returns (string)'
      ],
      provider
    );

    const [ethgrBalance, decimals, symbol] = await Promise.all([
      ethgrContract.balanceOf(ownerAddress),
      ethgrContract.decimals().catch(() => 18),
      ethgrContract.symbol().catch(() => 'TOKEN')
    ]);

    console.log(`💰 ${symbol} Balance:`, ethers.formatUnits(ethgrBalance, decimals), symbol);
  } else {
    console.log('❌ Private key does not match owner wallet!');
    console.log('💡 Need to find the correct private key for:', ownerAddress);
  }
}

main()
  .then(() => {
    console.log('\n✅ Verification completed!');
    process.exit(0);
  })
  .catch((error) => {
    console.error('❌ Verification failed:', error);
    process.exit(1);
  });