from quantkit.portal.flexi.api import *
from quantkit.utils.charting import *

from datetime import date as date
from ipywidgets import widgets
from IPython.display import display, clear_output
import pandas as pd
from IPython.html import widgets
from quantkit.portal.flexi.database import *
from quantkit.utils.ui_helpers import *

import math
import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets
import numpy as np
import os
from eq_analyzer import *
from quantkit.utils.excel import *

asm = get_master()

class MarketMonitoring():  
    
    
    def draw_top_dashboard(self):
        
        
        def on_dates_chg(dt_start,dt_end):
            self.start_date = dt_start
            self.end_date = dt_end
            pass
        
        w_dr = UIDateRange(on_dates_chg, 'settings/date_settings.pk')
        
        
        #Sets the start and end date from the date range ui
        self.start_date = w_dr.dt_start
        self.end_date = w_dr.dt_end
        w_dr.draw()
        
        
        geo={'DM':['N America', 'Europe', 'DM Asia'],'EM':['EM Asia', 'EMEA', 'LatAm']}

        geo2={'N America':['US', 'Canada'],
              'Europe':['Euro Zone', 'Germany', 'France', 'Italy', 'Spain', 'Netherlands',
               'Portugal', 'Ireland', 'Austria', 'Belgium', 'UK', 'Switzerland',
               'Sweden', 'Norway', 'Denmark', 'Finland', 'Israel'],
              'DM Asia':['Japan', 'Hong Kong', 'Singapore', 'Australia', 'New Zealand'],
              'EM Asia':['Asia ex', 'China', 'S Korea', 'Taiwan', 'India', 'Indonesia',
               'Malaysia', 'Philippines', 'Thailand', 'Vietnam', 'Pakistan'],
              'EMEA':['EMEA', 'Russia', 'S Africa', 'Turkey', 'Poland', 'Czech Rep',
               'Hungary', 'Romania', 'Greece', 'Egypt', 'Saudi Arabia', 'UAE',
               'Qatar'],
              'LatAm':['LatAm', 'Brazil', 'Mexico', 'Chile', 'Argentina', 'Peru',
               'Colombia'],

             }
        
        def print_region(region,country):
            self.country = self.country2.value
            print(self.country)
            pass

        def select_city(state):
            cityW.options = geo[state]

        #add in 'select district' function that looks in the new dictionary
        def select_country(region):
            self.country2.options = geo2[region]
            

        scW = widgets.Dropdown(options=geo.keys())
        init = scW.value
        cityW = widgets.Dropdown(options=geo[init])

        init2= cityW.value #new start value for district dropdown
        self.country2 = widgets.Dropdown(options=geo2[init2]) #define district dropdown widget
        self.country = self.country2.value
       

        dd_state = widgets.interactive(select_city, state=scW)
        dd_region = widgets.interactive(print_region, region=cityW, country=self.country2) #define district value
        dd_country2 = widgets.interactive(select_country, region=cityW) #call everything together with new interactive
        dd_chart_type = widgets.Dropdown(
            options=['Spread','Historical Yield Curve','Real Yield','10Y Attribution'],
            value='Historical Yield Curve',
            description='Type of Chart',
        )
        shared_box = widgets.HBox([w_dr.box,
                                   widgets.VBox([dd_state,dd_region])])
        return shared_box   
    
    def draw_dm(self):
        dm_fig = plt.figure(figsize=(24,12))
        dm_output = widgets.Output(layout={'border': '1px solid black'}) 
        btn_dm = widgets.Button(description="Generate DM")
        
        
        wd_path = os.getcwd() + '\DM EM Macro.xlsx'
        data = pd.read_excel(wd_path,'master')
        
        import math


        def select_data(str_cty,str_sector,str_category):
            df = data[data['Country'] == str_cty] # Select Cty
            df = df[df['Sector'] == str_sector] # Select Sector
            if str_category != 'All':
                df = df[df['Category'] == str_category] #Select Cat
            elif str_category == 'All':
                pass
            return df

        def plot_data(str_cty,str_sector,str_category,dt_start,dt_end): 

            df = select_data(str_cty,str_sector,str_category)

            if str_category != 'All':            
                lst_ticker = df['Ticker']
                lst_ticker = lst_ticker.reset_index()

                df_plot = pd.DataFrame(create_series(lst_ticker.loc[0]['Ticker']).data)


                for ticker in range(1,len(lst_ticker)):
                    df_plot[lst_ticker.loc[ticker]['Ticker']] = create_series(lst_ticker.loc[ticker]['Ticker']).data
 

                int_threshold = select_data(str_cty
                                                 ,str_sector
                                                 ,str_category)['Threshold']
                int_threshold = int_threshold.reset_index()
                int_threshold = int_threshold.loc[0]['Threshold']


                df_plot['Threshold'] = int_threshold
                df_plot = df_plot[dt_start:dt_end]
                return df_plot

            elif str_category == 'All':         
                return df

        
        def dm_btn_pressed(b):
            dm_fig.clear()
            dm_output.clear_output()


            #Pulls data using plot_data
            df = plot_data(self.country,
                           sectorSelector.value,
                           cat2.value,
                           self.start_date,
                           self.end_date)
            
            
            #Remove any tickers not included in the database
            try:
                df = df[df['Ticker']!= 'FiscalMonitorDatabase']
                df = df[df['Ticker']!= 'MSM1KE Index']
            except:
                pass

            with dm_output:
                print('Loading....')

                #Checks if user wants all the charts for the sector. 
                if cat2.value == "All":
                    try:
                        dm_fig.suptitle(self.country + " " + sectorSelector.value 
                                        + '\n ' + self.start_date.strftime('%d-%b-%Y') 
                                        + ' to ' + self.end_date.strftime('%d-%b-%Y'),
                                        fontsize=20, 
                                        fontweight='bold', 
                                        y=1.05)
                    except:
                        print(self.country + ' is not covered in the database!')
                    
                    df1 = df[['Category','Ticker','Threshold','Order','Kind','MA','Percent']]
                    dict1 = df1.set_index('Category').T.to_dict('list')
                    row = math.ceil(len(df1['Category'])/3)
                    col = 1

                    for key in range(len(df1['Category'])):
                        #Checking for MA
                        
                        ma = 0 
                        percent = 0
                        
                        #Setting the Ma & percent
                        ma = df1.iloc[key,5]
                        percent = df1.iloc[key,6]
                        
                        #Setting the Category
                        cat = df1.iloc[key,0]
                        
                        if cat == df1.iloc[key-1,0]:
                            col -=1
                            pass
                        else:
                            ax = dm_fig.add_subplot(row,3,col)
                        ticker = df1.iloc[key,1]  
                        df_plot = create_series(ticker).data           
                        df_plot = df_plot[self.start_date:self.end_date]
                        df_plot = pd.DataFrame(df_plot)                

                        #Inserting Threshold into the df
                        threshold = df1.iloc[key,2]

                        try:
                            if cat != df1.iloc[key-2,0]:
                                if not math.isnan(threshold):                    
                                    df_plot['Threshold'] = threshold
                        except:
                            pass

                        #Plots the Threshold Lines
                        try:
                            df_plot['Threshold'].plot(ax = ax, 
                                         title = sectorSelector.value + ' - ' + df1.iloc[key,0],
                                         figsize = (20,15),
                                        style = ['--'],
                                        color='green')
                            df_plot = df_plot.drop('Threshold',axis=1)
                        except:
                            pass                

                        #Plots the BM lines
                        try:
                            try:
                                consensus_df = create_series(ticker, 'BN_SURVEY_MEDIAN').data
                                consensus_latest = consensus_df.iloc[-1]
                                df_plot.columns = [df_plot.columns[0] 
                                                   + '\n C=' + str(consensus_latest) 
                                                   + '\n P=' + str(df_plot.iloc[-2,0])
                                                   + '\n A=' +  str(df_plot.iloc[-1,0])]
                            except:
                                df_plot.columns = [df_plot.columns[0]                                               
                                                   + '\n P=' + str(df_plot.iloc[-2,0])
                                                   + '\n A=' +  str(df_plot.iloc[-1,0])]
                                print(ticker + " consensus not available!")
                            

                            #Adjusting for MA
                            if not math.isnan(ma):
                                print('Non 0 ma = ' + str(ma))
                                
                                #need to convert float ma to int for rolling function
                                ma = int(ma)
                                
                                df_plot.iloc[:,0] = df_plot.iloc[:,0].rolling(ma,min_periods= 1).mean()

                                print('rolling done')
                            if not math.isnan(percent):
                                if percent == 12:
                                    print('YOY')
                                    past_df = create_series(ticker).data
                                    past_start = self.start_date -pd.DateOffset(years=1)
                                    past_end = self.end_date -pd.DateOffset(years=1)
                                    past_df = past_df[past_start:past_end]
                                    past_df = past_df.iloc[-len(df):]
                                    for pos in range(len(df)):
                                        df.iloc[pos,:] = df.iloc[pos,:] / past_df[pos]
                                    print('YOY')
                                elif percent == 4:                                    
                                    qoq_start = self.start_date -pd.DateOffset(months=3)
                                    past_df = create_series(ticker).data
                                    past_df = past_df[qoq_start:self.end_date]
                                    for pos in range(len(df_plot)):
                                        df_plot.iloc[pos,0] = df_plot.iloc[pos,0] / past_df[pos] 
                                    print('QOQ')


                            df_plot.plot(ax = ax, 
                                         title = sectorSelector.value + ' - ' 
                                         + df1.iloc[key,0],
                                         figsize = (20,15))
                            for line in ax.get_lines():
                                line.set_linewidth(1.5)

                            col += 1

                        except:
                            print('Error with '+ str(ticker)+ ',  Category: ' + cat)
                            col += 1
                            continue            
                    dm_fig.tight_layout(pad=2)
                    save_path = os.getcwd() + '\Market_Monitoring_Results\DM_plots_'
                    dm_fig.savefig(save_path + self.country + '_' 
                                   + sectorSelector.value + '_'
                                   + str(date.today()), 
                                   bbox_inches='tight', 
                                   pad_inches=0.25)
                    display(dm_fig)
                else:
                    dm_fig.clear()        
                    ax3 = dm_fig.add_subplot(111)
                    ticker = df.columns[0]
                    
                    
                    #checking MA / Percent
                    fields_df = select_data(self.country,
                                            sectorSelector.value,
                                            cat2.value)
                    ma = 0
                    percent = 0 
                    
                    #Assign MA
                    ma = fields_df['MA']
                    percent = fields_df['Percent']

                        
                    try:
                        #Plots the Threshold Lines

                        df['Threshold'].plot(ax = ax3,
                                figsize = (12,8),color='green',
                                style = ['--'])
                        df = df.drop('Threshold',axis = 1)
                        print("Plotted threshold.")

                        #Adds in the P/C/A

                        try: 
                            consensus_df = create_series(ticker, 'BN_SURVEY_MEDIAN').data

                            consensus_latest = consensus_df.iloc[-1]
                            df.columns = [df.columns[0] 
                                      + '\n C=' + str(consensus_latest) 
                                      + '\n P=' + str(df.iloc[-2,0])
                                      + '\n A=' + str(df.iloc[-1,0])]
                            print('Pulled consensus data.')
                        except:                           

                            if len(df.columns) == 2:
                                print('2 Indexes')  
                                df = df.rename(columns = {df.columns[0]: df.columns[0] + '\n P=' + str(df.iloc[-2,0]) + '\n A=' + str(df.iloc[-1,0]), 
                                                          df.columns[1]:  df.columns[1] + '\n P=' + str(df.iloc[-2,1]) + '\n A=' + str(df.iloc[-1,1])}, 
                                                           inplace = False)


                                
                            else:
                            
                                df.columns= [df.columns[0] 
                                          + '\n P=' + str(df.iloc[-2,0])
                                          + '\n A=' + str(df.iloc[-1,0])]

                            print("Consensus not available.")



                        #Adjusting Ma and percent
                        
                        print('Checking MA')
                        if len(ma) > 1:
                            print('More than 1 MA')
                            
                        else:
                            if not math.isnan(ma):
                                print('MA: ' + str(ma) )
                                df.iloc[:,0] = df.iloc[:,0].rolling(ma,min_periods = 1).mean()
                                print('Rolling calculations done.')

                            print('Checking percent')
                            if not math.isnan(percent):
                                if percent ==12:
                                    print('YOY')
                                    past_df = create_series(ticker).data
                                    past_start = self.start_date -pd.DateOffset(years=1)
                                    past_end = self.end_date -pd.DateOffset(years=1)
                                    past_df = past_df[past_start:past_end]
                                    past_df = past_df.iloc[-len(df):]
                                    for pos in range(len(df)):
                                        df.iloc[pos,:] = df.iloc[pos,:] / past_df[pos]
                                    print("YOY calculations done.")
                                elif percent == 4:
                                    print('QOQ')
                                    qoq_start = self.start_date -pd.DateOffset(months=6)
                                    past_df = create_series(ticker).data
                                    past_df = past_df[qoq_start:self.end_date]
                                    for pos in range(len(df)):
                                        df.iloc[pos,:] = df.iloc[pos,:] / past_df[pos-1]
                                    print('QOQ calculations done.')
                                

                        df.plot(ax = ax3,
                                figsize = (12,8),
                                title = self.country + ' '
                                        + sectorSelector.value + ' ' 
                                        + cat2.value)
                        print('Plotting done. ')
                            

                        #Saves the Fig
                        save_path = os.getcwd() + '\Market_Monitoring_Results\DM_plots_single_'
                        dm_fig.savefig(save_path 
                                       + self.country
                                       + '_' + sectorSelector.value 
                                    #    + '_' + cat2.value 
                                       + '_' + str(date.today()) ,
                                       bbox_inches='tight', pad_inches=0.25)
                        
                        print('Fig saved.')
                        display(dm_fig)

                    except:
                        print("Index not found!")
                    
                print('Done!')
        btn_dm.on_click(dm_btn_pressed)
        

        sectorSelector = widgets.Dropdown(options=data[data['Country']==self.country]['Sector'].unique())
        cat2 = widgets.Dropdown(options=data[(data['Country']==self.country)&(data['Sector']==sectorSelector.value)]['Category'].unique()) 

        def print_sector(sector,category):
            pass

        def select_sector(country):
            sector_options = data[data['Country']==self.country]['Sector'].unique()
            sectorSelector.options = sector_options
            
        def select_cat(sector):
            cat_options = data[(data['Country']==self.country)&(data['Sector']==sectorSelector.value)]['Category'].unique()
            cat_options = np.append(cat_options,'All')
            cat2.options = cat_options

        dd_dm1 = widgets.interactive(select_sector, country=self.country2)
        dd_dm2 = widgets.interactive(print_sector, sector=sectorSelector, category=cat2) 
        dd_dm3 = widgets.interactive(select_cat, sector=sectorSelector) 

        dm_box = widgets.VBox([dd_dm2,btn_dm,dm_output ])
        
        return dm_box
    
    def draw_asset_monitoring(self):
        region_file = os.getcwd() + '\\Market_Monitoring_Files\\region_countries.xlsx'
        df_map = pd.read_excel(region_file, index_col=0)
        df_map = df_map.drop('RID',axis=1)
        asm = get_master()
        lst_locs = df_map.index
        
        w_btn = widgets.Button(description='Run')
        fig = plt.figure(figsize=(12,6))
        w_o = widgets.Output(layout={'border': '1px solid black'})
        
        def plot_loc(b):
            fig.clear()
            w_o.clear_output()

            s_ctry = self.country
            s_ccy = df_map.loc[s_ctry, 'Currency']
            fx = asm.get_asset(s_ccy)
            rates = asm.rates.get_index(s_ctry)
            eqy = asm.get_asset('MSCI '+s_ctry)

            fx.update(kind='series', field='spot')
            eqy.update(kind='series', field='px')
            rates.update(kind='series', field='10y')

            df = pd.DataFrame({'EQ':eqy.px, 'FX':fx.spot, 
                               '10Y':rates.get_rates('10Y', kind='market')}).ffill()
            df = df[str(self.start_date)[:10]:str(self.end_date)[:10]]

            with w_o:
                ax = fig.add_subplot(221)
                df['EQ'].plot(ax=ax,figsize = (12,8),  style='C0',title = 'MSCI')
                ax.set_ylabel('MSCI '+s_ctry, color='C0')

                ax1 = fig.add_subplot(222)
                df['10Y'].plot(ax=ax1,figsize = (12,8),  style='C1',title = "10Y Yield",grid=True)
                ax1.set_ylabel('10Y', color='C1')

                ax2 = fig.add_subplot(223)
                df['FX'].plot(ax=ax2,figsize = (12,8),  style='C2',title= "Currency",grid=True)

                plot_loc.eq = df['EQ']
                plot_loc.yield2 = df['10Y']
                plot_loc.fx = df['FX']

                if not fx.inverse_quoted:
                    ax2.invert_yaxis()
                ax2.set_ylabel(s_ccy + ('' if fx.inverse_quoted else ' (inverted)') , color='C2')
                ax2.spines["right"].set_position(("axes", 1.075))
                tkw = dict(size=4, width=1.5)
                ax.tick_params(axis='y', colors='C0', **tkw)
                ax1.tick_params(axis='y', colors='C1', **tkw)
                ax2.tick_params(axis='y', colors='C2', **tkw)
                ax1.grid(True); ax2.grid(True)
                fig.suptitle(s_ctry +' Monitor: %s'%(df.index[-1].strftime('%d-%b-%Y')), 
                             fontsize=16, fontweight='bold', y=1.01)
                fig.savefig('plots/monitor/'+s_ctry, bbox_inches='tight', pad_inches=0.25)
                fig.tight_layout(pad=2)
                display(fig)
        w_btn.on_click(plot_loc)
            
        def export_btn_pressed2(b):
            date_now = time.strftime("%Y-%m-%d %H%M%S")

            save_path = os.getcwd() + '\Market_Monitoring_Results\Asset Monitoring_'
            writer = pd.ExcelWriter(save_path + self.country +'_' + date_now +'.xlsx', engine='xlsxwriter')

            # Write each dataframe to a different worksheet.
            plot_loc.eq.to_excel(writer, sheet_name='EQ')
            plot_loc.yield2.to_excel(writer, sheet_name='Yield')        
            plot_loc.fx.to_excel(writer, sheet_name='FX')

            # Close the Pandas Excel writer and output the Excel file.
            writer.save()
            print('Asset Monitoring Data Exported!!')

        btn_export2 = widgets.Button(description="Export Data")
        btn_export2.on_click(export_btn_pressed2)

        
        w_box = widgets.HBox([w_btn])
        w_box = widgets.VBox([w_box,btn_export2,w_o])

        return(w_box)       
    
    def draw_covid_comparison(self):
        covid_mapper = os.getcwd() + '\\Market_Monitoring_Files\\Covid_Mapper.xlsx'
        df_covid_map = pd.read_excel(covid_mapper)
        df_covid_map = df_covid_map.set_index('Country')
        df_total_data = pd.read_csv(os.getcwd() + '\\Market_Monitoring_Files\\'+'owid-covid-data.csv',
                                    parse_dates = True,
                                    infer_datetime_format = True)
        df_total_data['Date'] = pd.to_datetime(df_total_data['date'])
        df_total_data = df_total_data.set_index('Date')
        sf = df_covid_map[df_covid_map['Group'] == 'DM']
        
        lst_dm = df_covid_map[df_covid_map['Group'] == 'DM'].index
        lst_dm2 = df_covid_map[df_covid_map['Group'] == 'DM2'].index
        lst_EU2 = df_covid_map[df_covid_map['Group'] == 'EU2'].index
        lst_brics = df_covid_map[df_covid_map['Group'] == 'BRICS'].index
        lst_em = df_covid_map[df_covid_map['Group'] == 'EM'].index
        lst_em2 = df_covid_map[df_covid_map['Group'] == 'EM2'].index
        
        dict_dm ={}
        dict_dm2 = {}
        dict_EU2={}
        dict_brics={}
        dict_em = {}
        dict_em2 = {}
        
        #Covid Update Button
        def covid_btn_update(b):
            print('Loading....')
            loader = Downloader('https://covid.ourworldindata.org/data/owid-covid-data.csv',r'C:\Users\a23441\Documents\Covid')
            print('Covid Update Complete!!')            
        btn_covid_update = widgets.Button(description="Update Covid")
        btn_covid_update.on_click(covid_btn_update)
        
        
        def create_dict(listo,dictionary):
            for country in listo:
                tf = df_total_data['location'] == country
                dictionary[country] = df_total_data[tf]                
        def fill_all_dict():
            create_dict(lst_dm,dict_dm)
            create_dict(lst_dm2,dict_dm2)
            create_dict(lst_EU2,dict_EU2)
            create_dict(lst_brics,dict_brics)
            create_dict(lst_em,dict_em)
            create_dict(lst_em2,dict_em2)
            
            self.dict_of_dict = {'dict_dm':dict_dm,'dict_dm2':dict_dm2,
                        'dict_EU2':dict_EU2,
                        'dict_brics':dict_brics,
                        'dict_em':dict_em,'dict_em2':dict_em2}
            
            self.dict_cty_cb = {}
            for cty in df_covid_map.index:
                self.dict_cty_cb[cty] = cty                           
        fill_all_dict()        
        
        
        #Checkboxes
        names = []
        checkbox_objects = []
        i='data'
        if i == 'data':
            for key in self.dict_cty_cb:
                checkbox_objects.append(widgets.Checkbox(value=False, description=key))
                names.append(key)

        arg_dict = {names[i]: checkbox for i, checkbox in enumerate(checkbox_objects)}
        ui = widgets.VBox(children=checkbox_objects)
        selected_data = []
        def select_data(**kwargs):
            selected_data.clear()

            for key in kwargs:
                if kwargs[key] is True:
                    selected_data.append(key)

            print(selected_data)
        selected = widgets.interactive_output(select_data, arg_dict)
        
        
        #DD for type of comparison
        lst_headers =df_total_data.columns
        dd_covid_headers = widgets.Dropdown(
            options=lst_headers,
            value='new_cases',
            description='Data Selection'
        )
        
        
        #Generate Comparison Chart
        btn_covid = widgets.Button(description="Covid")
        def covid_btn_pressed(b):
            covid_fig.clear()
            covid_output.clear_output()
            with covid_output:
                covid_fig.clear()
                print(selected_data)
                ax3 = covid_fig.add_subplot(111)
                ax3.legend(selected_data)
                df = covid_plot(selected_data,dd_covid_headers.value)
                df.plot(ax = ax3,figsize = (12,8), title = dd_covid_headers.value)
                
                save_path = os.getcwd() + '\Market_Monitoring_Results\Covid_Comparison_'
                covid_fig.savefig(save_path  
                                + str(date.today()) ,
                               bbox_inches='tight', pad_inches=0.25)
                print('Fig saved at ' + save_path)
                display(covid_fig)
        def covid_plot(lst,s_header,save=False,
                       soothing_period= 7,b_rebase = True):
            df = pd.DataFrame()
            start = dict_dm['United States'].index[0]
            end = dict_dm['United States'].index[len(dict_dm['United States'].index)-1]
            for cty in lst:
                cty_dict = df_covid_map.loc[cty,'Dict']
                cty_dict = self.dict_of_dict[cty_dict]        
                df[cty] = cty_dict[cty][s_header].rolling(soothing_period, 
                                                          min_periods=1).mean()[start:end]
                df = df.loc[self.start_date:self.end_date,:]
            return df
        btn_covid.on_click(covid_btn_pressed)
        
        
        #Plotting of charts
        covid_fig = plt.figure(figsize=(12,6))
        covid_output = widgets.Output(layout={'border': '1px solid black'}) 
        covid_box = widgets.VBox([btn_covid_update,
                                  btn_covid,
                                  dd_covid_headers,
                                  ui])
        
        covid_box = widgets.HBox([covid_box,covid_output])
        
        
        return(covid_box)
    
        
    def draw_covid_overview(self):    
        covid_mapper = os.getcwd() + '\\Market_Monitoring_Files\\Covid_Mapper.xlsx'
        df_covid_map = pd.read_excel(covid_mapper)
        df_covid_map = df_covid_map.set_index('Country')
        df_total_data = pd.read_csv(os.getcwd() + '\\Market_Monitoring_Files\\'+'owid-covid-data.csv',
                                    parse_dates = True,
                                    infer_datetime_format = True)
        df_total_data['Date'] = pd.to_datetime(df_total_data['date'])
        df_total_data = df_total_data.set_index('Date')
        
        covid_fig_overview = plt.figure(figsize=(12,6))
        covid_overview_output = widgets.Output(layout={'border': '1px solid black'}) 
        btn_covid2 = widgets.Button(description="Covid Overview")
        
        
        def create_cty_covid(s_cty):
            df = df_total_data[df_total_data['location']== s_cty] 
            df = df.ffill()
            return df
        
        #Covid Update Button
        def covid_btn_update(b):
            print('Loading....')
            loader = Downloader('https://covid.ourworldindata.org/data/owid-covid-data.csv',r'C:\Users\a23441\Documents\Covid')
            print('Covid Update Complete!!')            
        btn_covid_update = widgets.Button(description="Update Covid")
        btn_covid_update.on_click(covid_btn_update)
        
        def covid_btn_pressed2(b):
            date_now = time.strftime("%Y-%m-%d %H%M%S")
            covid_fig_overview.clear()
            covid_overview_output.clear_output()
            with covid_overview_output:
                covid_fig_overview.clear()
                if self.country2.value == 'US':
                    s_cty = 'United States'
                elif self.country2.value == 'UK':
                    s_cty = 'United Kingdom'
                else: 
                    s_cty = self.country2.value
                df_newCases_total_vacc,df_newV_cases,df_totalV,df_newV = covid_cty_stats(s_cty,
                                                                                         start_date = str(self.start_date)[:10],
                                                                                         end_date = str(self.end_date)[:10])

                ax = covid_fig_overview.add_subplot(221)
                df_newCases_total_vacc.New_Cases_per_Million.plot(ax=ax,figsize=(12,8),title = 'New Cases vs Total Vaccinations',legend=True)
                df_newCases_total_vacc.Total_Vaccinations_per_Hundred.plot(ax=ax,secondary_y=True, legend=True)
                ax.grid(True)
                ax.right_ax.grid(False)


                ax2 = covid_fig_overview.add_subplot(222)
                df_newV_cases.plot(ax=ax2, title = "New Vaccinations - Cases",legend= False)
                ax2.set_xlabel('')
                ax2.xaxis.set_ticklabels([])
                ax2.grid(True)


                ax3 = covid_fig_overview.add_subplot(223)
                df_totalV.plot(ax=ax3, title = "Total Vaccinations")

                ax4 = covid_fig_overview.add_subplot(224)    
                df_newV.new_vaccinations_smoothed.plot(ax=ax4, title = "New Vaccinations",legend=True)
                df_newV.new_vaccinations_smoothed_per_million.plot(ax=ax4, secondary_y=True,legend=True,grid=False)

                h1,l1 = ax4.get_legend_handles_labels()
                h2,l2 = ax4.right_ax.get_legend_handles_labels()
                ax4.legend(h1+h2,['new','million'],ncol=2)

                ax4.grid(True)
                ax4.right_ax.grid(False)

                covid_fig_overview.tight_layout(pad=0.5)
                covid_fig_overview.suptitle(s_cty + ' : ' + str(self.end_date)[:10],fontsize=16, fontweight='bold', y=1.05)
                save_path = os.getcwd() + '\Market_Monitoring_Results\Covid_Overview_'
                covid_fig_overview.savefig(save_path + s_cty + '  '+str(date_now))
                print('Graph image saved at ' + save_path)
                display(covid_fig_overview)
        

        def covid_cty_stats(s_cty,save=False,
                            soothing_period= 7,
                            start_date='2020-12-1',
                            end_date='2021-5-1'):
            cty_dict = df_covid_map.loc[s_cty,'Dict']
            cty_dict = self.dict_of_dict[cty_dict] 

            df = create_cty_covid(s_cty)
            df_newCases_total_vacc = pd.DataFrame()
            df_newCases_total_vacc['New_Cases_per_Million'] = df['new_cases_smoothed_per_million'].rolling(soothing_period, min_periods=1).mean()[start_date:end_date] 
            df_newCases_total_vacc['Total_Vaccinations_per_Hundred'] = df['total_vaccinations_per_hundred'].rolling(soothing_period, min_periods=1).mean()[start_date:end_date]

            df_newV_cases = pd.DataFrame()
            df_newV_cases['New_Vaccinations_Cases'] = df['new_vaccinations'].rolling(soothing_period, min_periods=1).mean()[start_date:end_date] - cty_dict[s_cty]['new_cases'].rolling(soothing_period, min_periods=1).mean()[start_date:end_date]


            df_totalV = pd.DataFrame()
            df_totalV['people_fully_vaccinated_per_hundred']= df['people_fully_vaccinated_per_hundred'].rolling(soothing_period, min_periods=1).mean()[start_date:end_date]
            df_totalV['people_vaccinated_per_hundred']= df['people_vaccinated_per_hundred'].rolling(soothing_period, min_periods=1).mean()[start_date:end_date]

            df_newV = pd.DataFrame()
            df_newV['new_vaccinations_smoothed']= df['new_vaccinations_smoothed'].rolling(soothing_period, min_periods=1).mean()[start_date:end_date]
            df_newV['new_vaccinations_smoothed_per_million']= df['new_vaccinations_smoothed_per_million'].rolling(soothing_period, min_periods=1).mean()[start_date:end_date]

            covid_cty_stats.df_newCases_total_vacc = df_newCases_total_vacc
            covid_cty_stats.df_newV_cases = df_newV_cases
            covid_cty_stats.df_totalV = df_totalV
            covid_cty_stats.df_newV = df_newV  


            return df_newCases_total_vacc,df_newV_cases,df_totalV,df_newV        
        btn_covid2.on_click(covid_btn_pressed2)
        
        
        def export_btn_pressed3(b):
            date_now = time.strftime("%Y-%m-%d %H%M%S")
            save_path = os.getcwd() + '\Market_Monitoring_Results\Covid_'
            writer = pd.ExcelWriter(save_path + self.country +'_' + date_now +'.xlsx', engine='xlsxwriter')

            # Write each dataframe to a different worksheet.
            covid_cty_stats.df_newCases_total_vacc.to_excel(writer, sheet_name='New Cases to Total Vacc')
            covid_cty_stats.df_newV_cases.to_excel(writer, sheet_name='New cases')        
            covid_cty_stats.df_totalV.to_excel(writer, sheet_name='Total Vaccinations')
            covid_cty_stats.df_newV.to_excel(writer, sheet_name='New Vaccinations')

            # Close the Pandas Excel writer and output the Excel file.
            writer.save()
            print('Covid Data Exported!!')
        btn_export3 = widgets.Button(description="Export Covid")
        btn_export3.on_click(export_btn_pressed3)
        
        def plot_covid_bar(b):
            lst_all = ['United States','United Kingdom',
                       'European Union','Japan','Germany',
                       'France','Italy','Spain','Canada',
                        'Australia','Singapore','Brazil',
                       'Russia','India','China','South Africa',
                       'Indonesia','Malaysia','Thailand',
                       'Philippines','Turkey','Poland',
                       'Mexico','Chile']
            df_ppl_vaccinated = df_total_data.pivot(index='date',columns='location',values = 'people_vaccinated_per_hundred')
            df_ppl_vaccinated = df_ppl_vaccinated[lst_all]
            df_ppl_vaccinated = df_ppl_vaccinated.ffill()

            df_ppl_fully = df_total_data.pivot(index='date',columns='location',values = 'people_fully_vaccinated_per_hundred')
            df_ppl_fully = df_ppl_fully[lst_all]
            df_ppl_fully = df_ppl_fully.ffill()

            df_first_dose = df_ppl_vaccinated - df_ppl_fully
            df_bar = pd.DataFrame()
            df_bar['Second Dose'] = df_ppl_fully.loc['2021-05-17']
            df_bar['First Dose'] = df_first_dose.loc['2021-05-17']

            covid_fig_overview.clear()
            covid_overview_output.clear_output()

            with covid_overview_output:        

                ax = covid_fig_overview.add_subplot(211)
                ax.set_ylabel('May 2021')

                # plt.gca().axes.get_xaxis().set_visible(False)
                df_bar.plot(ax = ax,figsize = (12,8),kind='bar',stacked=True,grid = True)
                # ax.axes.get_xaxis().set_ticks([])
                ax.set_xlabel('')
                ax.xaxis.set_ticklabels([])


                ax2 = covid_fig_overview.add_subplot(212)
                ax2.set_ylabel('Difference vs April 21')
                df_bar2 = pd.DataFrame()
                df_bar2['Second Dose'] = df_ppl_fully.loc['2021-05-17']- df_ppl_fully.loc['2021-04-17']
                df_bar2['First Dose'] = df_first_dose.loc['2021-05-17']- df_first_dose.loc['2021-04-17']
                netchange = df_bar2['Second Dose'] + df_bar2['First Dose']
                df_bar2.plot(ax=ax2,figsize = (12,8),kind='bar',stacked=True)

                display(covid_fig_overview)
        btn_covid_bar = widgets.Button(description="1st vs 2nd")
        btn_covid_bar.on_click(plot_covid_bar)

        covid_overview_box = widgets.VBox([btn_covid_update,
                                           btn_covid2,
                                           btn_covid_bar,
                                           btn_export3,
                                           covid_overview_output])        
        return(covid_overview_box)        
        
    def draw_rates(self):
        df_info = pd.read_excel("DM EM Macro.xlsx")
        from datetime import date as date1
        rates = asm.rates
        
        
        def clean_rates(s_tenor,s_country):
            rates_index = rates.get_index(s_country)
            df_rates = rates_index.get_rates(s_tenor)
            dateo = date.today()
            dateo = dateo.strftime("%Y-%m-%d")
            idx = pd.date_range('1954-01-04', str(dateo))
            df_rates = df_rates.reindex(idx,fill_value = np.NaN)
            df_rates = df_rates.fillna(method='ffill')
            return df_rates

        def cls():
            w_o.clear_output()
            fig.clear()
            
        def hist_yields(s_country):
            #creates dictionaries
            dict_rates_now = {}
            dict_rates_1m = {}
            dict_rates_3m = {}
            dict_rates_6m = {}
            dict_rates_1y = {}

            #Sets up variables
            lst_duration = ['1M','2M','3M','6M','1Y','2Y','3Y','5Y','7Y','10Y','20Y','30Y']
            start_date = date1.today()
            print(start_date)
            one_month = start_date + relativedelta(months=-1)
            three_month = start_date + relativedelta(months=-3)
            six_month = start_date + relativedelta(months=-6)
            one_year = start_date + relativedelta(months=-12)

            lst_date = [start_date,
                        one_month,
                        three_month,
                        six_month,
                        one_year]
            rates_cty = rates.get_index(s_country)    


            final = pd.DataFrame()
            for date in lst_date:
                lst = []
                index = []
                for duration in lst_duration:
                    data = clean_rates(duration,s_country)
                    data_point = data[date]
                    data = data.dropna()
                    if len(data)==0:
                        continue
                    else:          
                        lst.append(data_point)
                        index.append(duration)
                lst = pd.DataFrame(lst)
                final = pd.concat([final,lst],axis=1)

            final.index = index
            final.columns = ['Current','1M','3M','6M','1Y']
            lst_pos = []
            for duration in final.index:
                if len(duration) == 2:
                    if duration[1] == "M":
                        lst_pos.append(int(duration[0])/12)
                    elif duration[1] == "Y":
                        lst_pos.append(int(duration[0]))
                elif len(duration) == 3:
                    lst_pos.append(int(duration[:2]))
            final['Duration'] = lst_pos
            return(final)

        output = widgets.Output()
        
        def generating_note():
            print("Charts Generated!")
        
        def generate_btn_pressed(b):
            fig.clear()
            output.clear_output()
            generate_btn_pressed.export = pd.DataFrame()
            with output:
                if dd_chart_type.value == 'Historical Yield Curve':
                    if self.country != 'All':
                        fig.clear()
                        with output:
                            ax = fig.add_subplot(111)
                            df = hist_yields(self.country)
                            df.plot(ax = ax,figsize = (12,8), 
                                    title = self.country + ' Historical Yields',
                                    x = df['Duration'],
                                    y=['Current','1M','3M','6M','1Y'])
                            ax.xaxis.set_ticks(df['Duration'])
                            ax.xaxis.set_ticklabels(df.index)
                            display(fig)

                elif dd_chart_type.value == 'Real Yield':
                    if self.country != 'All':
                        fig.clear()
                        with output:
                            ax = fig.add_subplot(111)
                            df=real_rates(self.country,
                                          '10Y',
                                          s_start_date =str(self.start_date)[:10],
                                          s_end_date =str(self.end_date)[:10] )
                            df.plot(ax = ax,figsize = (12,8),
                                    title = self.country + ' Real Yield')
                            display(fig)

                elif dd_chart_type.value == 'Spread':
                    if self.country != 'All':
                        fig.clear()
                        with output:
                            ax = fig.add_subplot(111)
                            df = spread(self.country,str(self.start_date)[:10])
                            df.plot(ax = ax,figsize = (12,8),
                                    title = self.country + ' 10-2Y Spread')
                            display(fig)
                            
                elif dd_chart_type.value == '10Y Attribution':
                    if self.country != 'All':
                        fig.clear()
                        with output:
                            ax = fig.add_subplot(111)
                            df = attribution10(s_country = self.country,
                               s_start_date = str(self.start_date)[:10],
                               s_end_date =str(self.end_date)[:10])
                            df.plot(ax = ax,figsize = (12,8),
                                    title = self.country+ ' 10Y Attribution')
                            display(fig)    
                generate_btn_pressed.export = df

        btn_generate = widgets.Button(description="Generate Charts")
        btn_generate.on_click(generate_btn_pressed) 
        
        
        
        def overview_btn_pressed(b):
            fig.clear()

            with output:
                clear_output()        
                ax1 = fig.add_subplot(221)
                hist = hist_yields(self.country)
                hist.plot(ax=ax1,
                          figsize = (12,8),
                          title = ' Historical Yields',
                          x = hist['Duration'],
                          y=['Current','1M','3M','6M','1Y'])
                ax1.xaxis.set_ticks(hist['Duration'])
                ax1.xaxis.set_ticklabels(hist.index)
                ax1.margins(0.05)

                ax2 = fig.add_subplot(222)
                df_spread = spread(self.country,
                                   str(self.start_date)[:10])
                df_spread.plot(ax=ax2,
                               figsize = (12,8),
                               title = ' 10Y-2Y Spread Yields')

                ax3 = fig.add_subplot(223)
                df_real = real_rates(self.country,
                                     '10Y',
                                     s_start_date = str(self.start_date)[:10],
                                     s_end_date =str(self.end_date)[:10] )
                df_real.plot(ax=ax3,
                             figsize = (12,8),
                             title = ' Real Yields')

                ax4 = fig.add_subplot(224)
                df_attribution = attribution10(s_country = self.country,
                                       s_start_date = str(self.start_date)[:10],
                                       s_end_date = str(self.end_date)[:10])
                df_attribution.plot(ax=ax4,
                                    figsize = (12,8),
                                    title = ' Attribution')

                overview_btn_pressed.historical = hist
                overview_btn_pressed.spread = df_spread
                overview_btn_pressed.attribution = df_attribution
                overview_btn_pressed.real = df_real
                try:
                    fig.suptitle(self.country2.value 
                                 + "\n" + "Data updated ao " 
                                 + str(date_ao),
                                 fontsize=16, 
                                 fontweight='bold', 
                                 y=1.05)
                except: 
                    "Please update data!"

                fig.tight_layout(pad=2)
                display(fig)


        btn_overview = widgets.Button(description="Generate Overview")
        btn_overview.on_click(overview_btn_pressed)
        
        

        lst_country = df_info['Country'].unique()
        lst_country = lst_country.tolist()
        lst_country.append('All')

        lst_sector = df_info['Sector'].unique()
        lst_sector = lst_sector.tolist()
        lst_sector.append('All')

        dd_sector = widgets.Dropdown(
            options=lst_sector,
            value='Monetary',
            description='Sector',
        )

        dd_chart_type = widgets.Dropdown(
            options=['Spread',
                     'Historical Yield Curve',
                     'Real Yield',
                     '10Y Attribution'],
            value='Historical Yield Curve',
            description='Type of Chart',
        )
        

        #Update Button
        def update_status():
            asm = get_master()
            global date_ao 
            date_ao = date.today()
            print("Update Complete!")
            print(date_ao)

        def update_btn_pressed(b):   
            last_update = date.today()
            with output:
                clear_output()
                update_status()
        btn_update = widgets.Button(description="Update Data")
        btn_update.on_click(update_btn_pressed)
        


        def export_btn_pressed(b):
            date_now = time.strftime("%Y-%m-%d %H%M%S")
            save_path = os.getcwd() + '\Market_Monitoring_Results\Rates_'

            s_file_path = save_path + str(date_now) + '.xlsx'

            if generate_btn_pressed.export.size != 0:
                df = generate_btn_pressed.export
                df.to_excel (s_file_path, startrow = 0,startcol = 0)
                print('Chart Data Exported!')
            elif overview_btn_pressed.historical.size !=0:
                # Create a Pandas Excel writer using XlsxWriter as the engine.
                df = pd.DataFrame()
                today = date.today()
                # df.to_excel(r'C:\Users\a23441\Documents\quantkit3a\quantkit\apps\market_watch\data\yields\Overview Data_' + str(date.today())+'.xlsx')  
                save_path = os.getcwd() + '\Market_Monitoring_Results\Rates_Overview_'
                writer = pd.ExcelWriter(save_path + date_now +'.xlsx', engine='xlsxwriter')

                # Write each dataframe to a different worksheet.
                overview_btn_pressed.historical.to_excel(writer, sheet_name='Historical Yield Curve')
                overview_btn_pressed.spread.to_excel(writer, sheet_name='10Y-2Y Spread')
                overview_btn_pressed.attribution.to_excel(writer, sheet_name='Yield Attribution')
                overview_btn_pressed.real.to_excel(writer, sheet_name='Real Yield')

                # Close the Pandas Excel writer and output the Excel file.
                writer.save()
                print('Overview Data Exported!!')
            else:
                raise Exception('Please generate a chart again!')


            generate_btn_pressed.export = pd.DataFrame()
            overview_btn_pressed.historical = pd.DataFrame()
            overview_btn_pressed.spread = pd.DataFrame()
            overview_btn_pressed.real = pd.DataFrame()

        btn_export = widgets.Button(description="Export Data")
        btn_export.on_click(export_btn_pressed)
        
        fig = plt.figure(figsize=(12,8))
        fig.tight_layout(pad=3.0)
        
        def attribution10(s_country, 
                          s_start_date = '1995-01-01',
                          s_end_date = str(date.today())):

            rates_cty = rates.get_index(s_country)

            idx = pd.date_range(s_start_date, s_end_date)

            be = rates.get_breakeven('10Y', market=s_country)
            be = be.reindex(idx,fill_value = np.NaN)
            be = be.fillna(method='ffill')
            be = be.dropna()

            nominal = rates.get_rates('10Y', market=s_country)
            nominal = nominal.reindex(idx,fill_value = np.NaN)
            nominal = nominal.fillna(method='ffill')
            nominal = nominal.dropna()

            real = nominal - be
            real = pd.DataFrame(real)
            real.columns = ['Real']

            nominal = pd.DataFrame(nominal)
            be = pd.DataFrame(be)

            nominal.columns = [s_country + ' Nominal']
            be.columns = [s_country + ' Breakeven']

            nominal = nominal.join(be,how='outer')
            nominal = nominal.join(real,how='outer')
            return nominal
        
        dateo = date.today()
        dateo = dateo.strftime("%Y-%m-%d")
        def real_rates(s_country,
                       s_tenor,
                       s_start_date ='1995-01-01',
                       s_end_date = dateo):
            dateo = date.today()
            dateo = dateo.strftime("%Y-%m-%d")
    
            be = rates.get_breakeven(s_tenor, market=s_country)
            idx = pd.date_range(s_start_date, s_end_date)
            be = be.reindex(idx,fill_value = np.NaN)
            be = be.fillna(method='ffill')
            be = be.dropna()


            nominal = rates.get_rates(s_tenor, market=s_country)
            nominal = nominal.reindex(idx,fill_value = np.NaN)
            nominal = nominal.fillna(method='ffill')
            nominal = nominal.dropna()
    
            real = nominal - be
            return real
        def spread(s_country,
                   start_date='2015-01-01'):
            rates = asm.rates
            rates_cty = rates.get_index(s_country)
            rates_cty.update()
            df_2Y = rates_cty.get_rates('2Y',start = start_date)
            df_10Y = rates_cty.get_rates('10Y',start= start_date)
            df_spread = df_10Y - df_2Y
            return df_spread
        
        fig = plt.figure(figsize=(12,8))
        fig.tight_layout(pad=3.0)
        
        
        left_box = widgets.VBox([btn_export,
                                 btn_generate,
                                 btn_overview,
                                 output])
        

        yield_box = widgets.HBox([left_box,
                                  dd_chart_type])
        
        return yield_box
        
    
    def draw_earnings(self):
        date_now = time.strftime("%Y-%m-%d %H%M%S")
        c_eq_analyzer = EquityAnalyzer(self.country, 'Asia ex', path='plots/indo')
        bm_path = os.getcwd() + '\\Market_Monitoring_Files\\MSCI.xlsx'
        msci = pd.read_excel(bm_path,'Universe')       
        
        
        dd_bm = widgets.Dropdown(description='Benchmark:',
                                 options=msci['Name'])
        
        returns_fig = plt.figure(figsize=(24,12))
        returns_output = widgets.Output(layout={'border': '1px solid black'}) 
        
        btn_returns_update = widgets.Button(description="Update")
        btn_returns_earnings = widgets.Button(description="Earnings")
        btn_returns_perf = widgets.Button(description="Performance")
        btn_returns_valuation = widgets.Button(description="Valuation")
        btn_returns_flows = widgets.Button(description="Flows")
        
        
        def returns_update_pressed(b,force = 'auto'):
            returns_fig.clear()
            returns_output.clear_output()
            c_eq_analyzer = EquityAnalyzer(self.country, dd_bm.value, path='plots/indo')
            with returns_output:
                print('Updating tr,fpe,pb,roe.... (1/3)')
                
            for s_series in ['tr', 'fpe', 'pb', 'roe']:
                c_eq_analyzer.eq_bm.update(kind='series', field=s_series, force=force)
                c_eq_analyzer.eq_x.update(kind='series', field=s_series, force=force)
            
            with returns_output:
                print('Updating indexes.... (2/3)')
            c_eq_analyzer.eq_x.index.update(field='BEST_EPS_BF', force=force)
            c_eq_analyzer.eq_x.index.update(field='BEST_EPS_1GY', force=force)
            c_eq_analyzer.eq_x.index.update(field='BEST_EPS_2GY', force=force)
            
            with returns_output:
                print('Updating macro....(3/3)')
            c_eq_analyzer.macro.update(kind='Portfolio Flow', force=force)
            
            with returns_output:
                print('Update Complete!')
            
        btn_returns_update.on_click(returns_update_pressed)
        
        def returns_earnings_pressed(b):
            returns_fig.clear()
            returns_output.clear_output()
            c_eq_analyzer = EquityAnalyzer(self.country, dd_bm.value, path='plots/indo')
            # sector f12
            df_em = c_eq_analyzer.eq_x.index.get_feps(currency='LOC').pct_change(63).last('36M').dropna(how='all', axis=1)
            df_em = df_em.rename(columns={k:k+'(%0.1f%%)'%(v*100) 
                for k, v in df_em.iloc[-1].iteritems() if not np.isnan(v)})
            # benchmark yearly
            df_worm = c_eq_analyzer.eq_x.index.get_feps(kind='worm',sector='Benchmark', 
                currency='LOC').last('36M').dropna(how='all', axis=1)
            ds_score= pd.Series({s_yr2:ts2.dropna()[-1]/ts1.dropna()[-1]-1 
                for (s_yr1, ts1), (s_yr2, ts2) in zip(df_worm.iloc[:,:-1].iteritems(), 
                df_worm.iloc[:,1:].iteritems())})
            df_worm = df_worm.rename(columns={k:'%d (%0.1f%%)'%(k, v*100) 
                for k, v in ds_score.iteritems() if not np.isnan(v)})
            df_worm['Forward 12M'] = c_eq_analyzer.eq_x.index.get_feps(kind='f12',
                sector='Benchmark', currency='LOC')
            
            with returns_output:
                returns_fig.subplots_adjust(hspace=0.05)
                ax = returns_fig.add_subplot(211)
                df_worm.plot(ax=ax)
                hide_xtick_labels(ax)
                ax.set_xlabel('')
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                ax = returns_fig.add_subplot(212)
                _, cbar = heatmap(df_em, ax=ax, cmap='jet_r', vmin=-0.1, vmax=0.1, 
                    cbar_kw={"orientation":"horizontal", "pad":0.15}, 
                    major_loc=YearLocator(), major_fmt=DateFormatter('%Y'))
                cbar.set_ticks(cbar.get_ticks())
                cbar.set_ticklabels(['%0.1f%%'%(x*100) for x in cbar.get_ticks()])
                returns_fig.suptitle(c_eq_analyzer.market + ' Equity Earnings: %s'%(df_worm.index[-1].strftime('%d-%b-%Y')), 
                             fontsize=16, fontweight='bold', y=0.925)

                save_path = os.getcwd() + '\Market_Monitoring_Results\Earnings_'
                returns_fig.savefig(save_path + c_eq_analyzer.market + '_earnings_'+str(date_now))
                
                display(returns_fig)
                
        btn_returns_earnings.on_click(returns_earnings_pressed)
        
        
        def returns_perf_pressed(b):
            returns_fig.clear()
            returns_output.clear_output()
            c_eq_analyzer = EquityAnalyzer(self.country, dd_bm.value, path='plots/indo')
            def get_tr(eq):
                df_tr = pd.DataFrame({'TR_'+s_ccy:eq.get_tr(currency=s_ccy)
                    for s_ccy in ['LOC', 'USD']}).ffill().dropna(how='all')
                return df_tr

            df_tr_id = get_tr(c_eq_analyzer.eq_x)
            df_tr_axj = get_tr(c_eq_analyzer.eq_bm)
            pn_tr = pd.Panel({c_eq_analyzer.market:df_tr_id, 
                c_eq_analyzer.benchmark:df_tr_axj}).ffill(axis='major')
            pn_tr[c_eq_analyzer.ratio] = pn_tr[c_eq_analyzer.market]/pn_tr[c_eq_analyzer.benchmark]
            df_tr = pn_tr.swapaxes('items', 'major').to_frame(False).T.ffill()
            df_tr.tail()


            returns_fig.subplots_adjust(hspace=0.2)
            def plot_one(df_trx, ax, title=None):
                df_trx = df_trx/df_trx.iloc[0]-1
                df_trx.plot(ax=ax)
                ax.plot(ax.get_xlim(), [0,0], 'r--')
                percentage_tick_labels(ax)
                ax.legend(['LOC', 'USD'], ncol=2)
                ax.yaxis.tick_right()
                ax.set_xlabel('')
                if title is not None:
                    ax.set_title(title)
            with returns_output:
                ds_per = pd.Series({'3M':'3M', '1Y':'12M', '3Y':'36M'}, 
                    index=['3Y', '1Y', '3M'])
                for i, (s_label, s_per) in enumerate(ds_per.iteritems()):
                    ax = returns_fig.add_subplot(2,3,i+1)
                    df_trx = df_tr[c_eq_analyzer.market].last(s_per)
                    plot_one(df_trx, ax, s_label)
                    if i == 0:
                        ax.set_ylabel(c_eq_analyzer.market)
                for i, (s_label, s_per) in enumerate(ds_per.iteritems()):
                    ax = returns_fig.add_subplot(2,3,i+4)
                    df_trx = df_tr[c_eq_analyzer.ratio].last(s_per)
                    plot_one(df_trx, ax)
                    if i == 0:
                        ax.set_ylabel(c_eq_analyzer.ratio)
                
                returns_fig.suptitle('%s Equity Performance: %s'%(c_eq_analyzer.market, 
                    df_tr.index[-1].strftime('%d-%b-%Y')),
                    fontsize=16, fontweight='bold', y=0.95)
                
                
                save_path = os.getcwd() + '\Market_Monitoring_Results\Earnings_'
                returns_fig.savefig(save_path + c_eq_analyzer.market + '_performance_'+str(date_now))

                display(returns_fig)
         
        btn_returns_perf.on_click(returns_perf_pressed)        
        
        def returns_valuation_pressed(b):
            returns_fig.clear()
            returns_output.clear_output()
            c_eq_analyzer = EquityAnalyzer(self.country, dd_bm.value, path='plots/indo')
            def get_fv(eq):
                df = pd.DataFrame({'P/Ef':eq.fpe, 'P/B':eq.pb, 'ROE':(eq.roe)},
                    columns=['P/Ef', 'P/B', 'ROE']).ffill().dropna(how='all')
                return df
            def to_monthly(df):
                dfm = df.resample('M').last()
                return dfm

            df_fv_x = get_fv(c_eq_analyzer.eq_x)
            df_fv_x_m = to_monthly(df_fv_x)
            df_fv_bm_m = to_monthly(get_fv(c_eq_analyzer.eq_bm))
            pn_fv = pd.Panel({c_eq_analyzer.market:df_fv_x_m, 
                              c_eq_analyzer.benchmark:df_fv_bm_m})\
                                .ffill(axis='major').swapaxes('items', 'minor').iloc[:,-120:]

            def plot_simple_valuation(ts, inverse=False, tail=3, helper=None, 
                nfmt="{:.1f}", ax=None, figsize=(12,6)):
                if ax is None:
                    returns_fig, ax = plt.subplots(figsize=figsize)
                def calc_stats(ts):
                    ts = ts[ts>0]
                    f_avg, f_std = ts.mean(), ts.std()
                    return f_avg, f_std
                f_avg, f_std = calc_stats(ts)
                df = pd.DataFrame({'+1 Std':f_avg+f_std,
                                   'Avg':f_avg,
                                   '-1 Std':f_avg-f_std,
                                   'Actual':ts},
                    columns=['+1 Std', 'Avg', '-1 Std', 'Actual'])
                if inverse:
                    df.plot(ax=ax, style=['r', 'y', 'g', 'b'])
                else:
                    df.plot(ax=ax, style=['g', 'y', 'r', 'b'])
                annotate_time_series(nfmt.format(ts[-1]), 
                                     (ts.index[-1], ts[-1]), 
                                     ax)
                ax.legend(loc='upper left', 
                          ncol=4)
                y1, y2 = ax.get_ylim()
                y1 = y1 if y1 > f_avg-tail*f_std else f_avg-tail*f_std
                y2 = y2 if y2 < f_avg+tail*f_std else f_avg+tail*f_std
                ax.set_ylim(y1, y2)
                ax.yaxis.tick_right()
                f_zscore = (ts[-1]-f_avg)/f_std*(-1 if inverse else 1)
                if helper is None:
                    s_label = 'z=%0.2f'%(f_zscore)
                else:
                    s_label = helper + ' (z=%0.2f)'%(f_zscore)
                ax.set_ylabel(s_label)
                ax.set_xlabel('')

                return ax

            with returns_output:
                returns_fig.subplots_adjust(hspace=0.05)
                dc_inv = {s_series:True if s_series!='ROE' else False 
                          for s_series in ['P/Ef', 'P/B', 'ROE']}
                for i, (s_series, df) in enumerate(pn_fv.iteritems()):
                    ax = returns_fig.add_subplot(3,2,i*2+1)
                    plot_simple_valuation(df[c_eq_analyzer.market], 
                                          inverse=dc_inv[s_series], 
                                          helper=s_series, 
                                          ax=ax)
                    if i == 0:
                        ax.set_title(c_eq_analyzer.market)
                    ax = returns_fig.add_subplot(3,2,i*2+2)
                    plot_simple_valuation(df[c_eq_analyzer.market]/df[c_eq_analyzer.benchmark], 
                        inverse=dc_inv[s_series], helper=s_series, ax=ax)
                    if i == 0:
                        ax.set_title(c_eq_analyzer.ratio)
                for i, ax in enumerate(returns_fig.axes):
                    if i < 4:
                        hide_xtick_labels(ax)
                returns_fig.suptitle(c_eq_analyzer.market + ' Equity Valuation: %s'%(df_fv_x.index[-1].strftime('%d-%b-%Y')),
                    fontsize=16, fontweight='bold', y=0.925)
        
                save_path = os.getcwd() + '\Market_Monitoring_Results\Earnings_'
                returns_fig.savefig(save_path + c_eq_analyzer.market + '_valuation_'+str(date_now))

                display(returns_fig)
            
        btn_returns_valuation.on_click(returns_valuation_pressed)
        
        def returns_flows_pressed(b):
            returns_fig.clear()
            returns_output.clear_output()
            c_eq_analyzer = EquityAnalyzer(self.country, dd_bm.value, path='plots/indo')
            ts_eflow = c_eq_analyzer.macro.get_portfolio_flow(asset='equity').iloc[:,0]
            ts_eflow = ts_eflow.resample('B').last().ffill()

            with returns_output:
                returns_fig.subplots_adjust(hspace=0.05)
                ax = returns_fig.add_subplot(211)
                ts_eflowx = ts_eflow.last('60M')
                ts_eflowx -= ts_eflowx.iloc[0]
                ts_eflowx.plot(ax=ax, kind='area', stacked=False)
                ax.set_xlabel(''); hide_xtick_labels(ax)
                ax.set_ylabel('5Y Cumu. Flow (US$/Bn)')
                ax = returns_fig.add_subplot(212)
                ts_eflow.diff(63).last('60M').plot(ax=ax, kind='area', stacked=False)
                ax.set_ylabel('Rolling 3M Flow (US$/Bn)')
                for ax in returns_fig.axes:
                    ax.yaxis.tick_right()
                returns_fig.suptitle(c_eq_analyzer.market + ' Foreign Equity Flow: %s'%(ts_eflow.index[-1].strftime('%d-%b-%Y')), 
                             fontsize=16, fontweight='bold', y=0.925)
                display(returns_fig)
                                     
                save_path = os.getcwd() + '\Market_Monitoring_Results\Earnings_'
                returns_fig.savefig(save_path + c_eq_analyzer.market + '_flows_'+str(date_now))

                display(returns_fig)       
            
        btn_returns_flows.on_click(returns_flows_pressed)
        
        earnings_buttons_box = widgets.VBox([dd_bm,
                                             btn_returns_update,
                                             btn_returns_earnings,
                                             btn_returns_perf,
                                             btn_returns_valuation,
                                             btn_returns_flows])
        
        earnings_box = widgets.HBox([earnings_buttons_box,
                                    returns_output])
        
        return earnings_box
            
            
    
    def draw_tabs(self):
        
        #Earnings Tab
        earnings = self.draw_earnings()
        
        
        #Research Tab
        dm_box = self.draw_dm()
        research_tab = widgets.Tab(children = [dm_box])
        research_tab.set_title(0, 'DM') 
        
        #Asset Monitoring Tab
        asset_monitoring = self.draw_asset_monitoring()

        #Rates Tab
        rates = self.draw_rates()
        rates_tab = widgets.Tab(children = [rates])
        
        
        #Covid Tab
        covid_comparison = self.draw_covid_comparison()
        covid_overview = self.draw_covid_overview()
        covid_tab = widgets.Tab(children = [covid_comparison,
                                                 covid_overview])
        covid_tab.set_title(0, 'Cty Comparison')
        covid_tab.set_title(1, 'Cty Overview')
        
        #Overall Tab        
        overall_tab = widgets.Tab(children = [research_tab,
                                              asset_monitoring,
                                              covid_tab,
                                              rates_tab,
                                              earnings,
                                             ])

        tab_names = ['Research Pack',
                     'Asset Monitoring',
                     'Covid',
                     'Yield',
                     'Earnings'
                    ]
        for pos in range(len(tab_names)):
            overall_tab.set_title(pos, tab_names[pos])
        print('Overall tab')
        return overall_tab   

    def draw_final(self):
        shared_box = self.draw_top_dashboard()     
        overall_tab = self.draw_tabs()
        final = widgets.VBox([shared_box,overall_tab])
        display(final) 
    