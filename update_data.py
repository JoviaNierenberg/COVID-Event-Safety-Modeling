import urllib3
import logging

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

with open('covidactnow_api.txt', 'r') as file:
    api_key = file.read().replace('\n', '')

logger = logging.getLogger('projEpsilon')
hdlr = logging.FileHandler('data_update.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

state_url = "https://api.covidactnow.org/v2/states.csv?apiKey="+api_key
us_timeseries_url = "https://api.covidactnow.org/v2/country/US.timeseries.csv?apiKey="+api_key
covidestim_url = "https://covidestim.s3.us-east-2.amazonaws.com/latest/state/estimates.csv"

state_file = "/Users/shagun/Downloads/states.csv"
us_timeseries_file = "/Users/shagun/Downloads/UStimeseries.csv"
covidestim_file = "/Users/shagun/Downloads/covidestim_estimates.csv"

with urllib3.PoolManager() as http:
    try:
        r = http.request('GET', state_url)
        logger.info('state data updated')
    except:
        logger.warning('unable to update state data')
    with open(state_file, 'wb') as fout:
        fout.write(r.data)
    
    try:
        r = http.request('GET', us_timeseries_url)
        logger.info('us timeseries data updated')
    except:
        logger.warning('unable to update us timeseries data')   
    with open(us_timeseries_file, 'wb') as fout:
        fout.write(r.data)
    
    try: 
        r = http.request('GET', covidestim_url)
        logger.info('covidestim data updated')
    except:
        logger.warning('unable to update covidestim data')
    with open(covidestim_file, 'wb') as fout:
        fout.write(r.data)