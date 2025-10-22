Important: This is not legal or tax advice. Verify with qualified professionals.

Scope: Trading US-listed equities from Australia (e.g., NASDAQ/NYSE) for your own account.

Residency and Accounts
- Broker access: Use a broker that supports Australian residents trading US markets (e.g., IBKR Australia, Stake, etc.).
- KYC/AML: Expect identity and residency verification.

Tax Forms and Withholding
- W-8BEN: Submit to your broker to certify non-US status. Reduces US dividend withholding (commonly 30% → 15%) under AU–US treaty. Renew periodically.
- Dividends: US withholding applied at source. Confirm correct treaty rate in your broker statements.
- Capital gains: Generally taxed in Australia per ATO rules. Keep accurate records in AUD (FX conversion). Consult a tax advisor.
- Wash sale rule: US wash-sale rule typically applies to US taxpayers; for Australian residents, check ATO guidance on loss deductions and anti-avoidance rules.

Market Data and Licensing
- Real-time data: US exchanges (NASDAQ/NYSE/OPRA) require market data agreements. You may need to self-certify as Non-Professional to access lower fees.
- Redistribution: Data provider ToS often prohibit redistribution/commercial use without a license. Ensure your use (incl. algo/ML storage) complies.
- Delayed vs. real-time: Some APIs allow delayed data for free. For real-time trading you likely need paid data via your broker or a vendor.

Trading Rules and Conduct
- Pattern Day Trader (PDT): FINRA PDT applies to accounts with < $25k that execute 4+ day trades in 5 business days; check your broker’s enforcement for non-US residents.
- Short selling: Uptick rules, borrow availability, and broker restrictions apply.
- Extended hours: Pre/post-market sessions have different liquidity/volatility and may require specific permissions.
- Market manipulation and spoofing: Strictly prohibited under US and AU law (SEC/FINRA/ASIC). Ensure your algo complies.

Australian Regulations
- Personal vs. business: If trading for others or providing advice/signals, you may need an Australian Financial Services Licence (AFSL) and must comply with ASIC rules.
- Record keeping: Maintain robust records (orders, fills, logs, model decisions) to substantiate tax and compliance inquiries.

Operational Considerations
- Timezone: US RTH roughly 23:30–06:00 (AEST) or 00:30–07:00 (AEDT), varying with DST on both sides.
- Latency/hosting: Consider running your algo on a server closer to your broker/exchange gateways. Mind data privacy and ToS.
- Disaster recovery: Implement fail-safes: max position caps, kill-switches, and monitoring.

Checklist (Non-exhaustive)
- Broker account approved for US equities + market data subscriptions active
- W-8BEN on file; understand dividend withholding and ATO reporting
- PDT and short-sale rules understood for your broker
- Data/API ToS reviewed; commercial use/licensing clarified
- Risk controls implemented (position limits, gross/net exposure, throttle)
- Logging and audit trail in place
- Backups and process monitoring configured
