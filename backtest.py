from lazyft.command_parameters import BacktestParameters

params = BacktestParameters(
    config_path="user_data/config.json",
    # timerange='20211121-20211212',
    # timerange='20210110-20210131',
    # timerange='20220102-20220116',
    timerange="20220101-",
    interval='1h',
    # days=365,
    stake_amount="unlimited",
    starting_balance=1000,
    max_open_trades=1,
    pairs=[
        'BTC/USDT',
        # 'ETH/USDT',
    ],
    download_data=True,
)
if __name__ == '__main__':
    runner = params.run('BatsContest', load_from_hash=False)
    report = runner.save()
    print(f'Sortino: {report.sortino_loss}', f'Sharpe: {report.sharp_loss}', sep='\n')
