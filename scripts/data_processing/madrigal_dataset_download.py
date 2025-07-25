from madrigalWeb import madrigalWeb
import datetime
import os

#create tec_dir and ne_dir if they do not exist
if not os.path.exists('tec_dir'):
    os.makedirs('tec_dir')
if not os.path.exists('ne_dir'):
    os.makedirs('ne_dir')

for year in range(2010, 2024):
    try:
        # Initialize Madrigal server
        madrigal_url = 'http://cedar.openmadrigal.org/'
        madweb = madrigalWeb.MadrigalData(madrigal_url)

        # Define the time range (datetime objects)
        start_time = datetime.datetime(int(year), 1, 1, 0, 0, 0)
        end_time = datetime.datetime(int(year+1), 12, 31, 23, 59, 59)

        # Get experiments using explicit date/time arguments
        experiments = madweb.getExperiments(
            0,  # 0 means all instruments
            start_time.year, start_time.month, start_time.day,
            start_time.hour, start_time.minute, start_time.second,
            end_time.year, end_time.month, end_time.day,
            end_time.hour, end_time.minute, end_time.second
        )

        electron_density=[]
        tec=[]
        for experiment in experiments:
            print(f"Experiment: {experiment.name},")
            files = madweb.getExperimentFiles(experiment.id)
            print(f"Found {len(files)} files for experiment {experiment.name}")
            for file in files:
                print(f"File ID: {file.expId}, Name: {file.name}")

                # Get parameters in this file
                params = madweb.getExperimentFileParameters(file.name)

                # params should be a list of parameter objects with name/description
                for p in params:
                    #print(p.description)
                    if p.mnemonic in 'TEC':
                        #print(p.description)
                        #'GDALT', 'GDLAT', 'GLON', 'TEC', 'NE', 'file_name', 'file_id']
                        #Year (universal time)
                        #Month (universal time)
                        #Day (universal time)
                        #Hour (universal time)
                        #Minute (universal time)
                        #Second (universal time)
                        tec.append((file.expId, file.name))
                        remote_path = file.name
                        local_filename = os.path.join('tec_dir',file.name.split('/')[-1])

                        madweb.downloadFile(
                            remote_path,
                            local_filename,
                            user_fullname='Giacomo Acciarini',
                            user_email='giacomo.acciarini@gmail.com',
                            user_affiliation='University of Surrey'
                        )

                    elif p.mnemonic in 'NE':
                        electron_density.append((file.expId, file.name))
                        remote_path = file.name
                        
                        local_filename = os.path.join('ne_dir',file.name.split('/')[-1])

                        madweb.downloadFile(
                            remote_path,
                            local_filename,
                            user_fullname='Giacomo Acciarini',
                            user_email='giacomo.acciarini@gmail.com',
                            user_affiliation='University of Surrey'
                        )
    except Exception as e:
        print(f"Error processing year {year}: {e}")
        #Electron density (NE)
        #TEC -> vertically integrated electron density
