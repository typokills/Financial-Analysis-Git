
import os
from IPython.display import display
import ipywidgets as widgets
from ipywidgets import Button, Layout
from datetime import datetime
import matplotlib.pylab as plt

import pandas as pd
import pandas.tseries.offsets as date_offsets
import numpy as np


def offset(dt, offset):
    dict_offset = {'Previous Close':date_offsets.BDay(),
        'End of Last Week':date_offsets.Week(weekday=4),
        'End of Last Month':date_offsets.BMonthEnd(),
        'End of Last Quarter':date_offsets.BQuarterEnd(),
        'End of Last Year':date_offsets.BYearEnd(),
        '1M Ago':date_offsets.BMonthEnd()*1,
        '3M Ago':date_offsets.BMonthEnd()*3,
        '6M Ago':date_offsets.BMonthEnd()*6,
        '1Y Ago':date_offsets.BMonthEnd()*12,
        '3Y Ago':date_offsets.BMonthEnd()*36,
        '5Y Ago':date_offsets.BMonthEnd()*60,
        '10Y Ago':date_offsets.BMonthEnd()*120,
        '15Y Ago':date_offsets.BMonthEnd()*180,
        '20Y Ago':date_offsets.BMonthEnd()*240,
        '30Y Ago':date_offsets.BMonthEnd()*360,
        '50Y Ago':date_offsets.BMonthEnd()*600}
    return dt - dict_offset[offset]

class UIDateRange(object):
    """docstring for UIDateRange"""
    def __init__(self, notify, path):
        super().__init__()
        self.fn_notify = notify

        settings = pd.Series({'Start_Date':pd.NaT,
            'Relative':True,
            'Relative_Opt':'3Y Ago',
            'End_Date':pd.NaT,
            'End_Opt':'Previous Close'})
        self.settings = SimpleSettings(settings, path=path)

        dt_today = pd.Timestamp(datetime.today().date())
        if self.settings.End_Opt == 'Custom':
            self.dt_end = self.settings.End_Date
        else:
            self.dt_end = offset(dt_today, self.settings.End_Opt)
        if self.settings.Relative_Opt == 'Custom':
            self.dt_start = self.settings.Start_Date
        else:
            self.dt_start = offset(self.dt_end, self.settings.Relative_Opt)

        self.dt_today = dt_today

        self.b_start_auto = False
        self.b_end_auto = False

    def draw(self):
        w_l11 = widgets.Label('Start Date')
        lo = Layout(width='80%', height='20px')
        w_l12 = widgets.Label('Relative to End Date', layout=lo)
        w_l21 = widgets.Label('End Date')
        w_l22 = widgets.Label('End Date Option', layout=lo)
        w_l_error = widgets.Label('', font_color = 'r',
            layout=lo)
        w_d1 = widgets.DatePicker(
            value=self.dt_start,
            description='',
            disabled=False,
            layout=lo
        )
        w_d2 = widgets.DatePicker(
            value=self.dt_end,
            description='',
            disabled=False,
            layout=lo
        )

        lst_relative = ['End of Last Week', 'End of Last Month', 
            'End of Last Quarter', 'End of Last Year', '1M Ago',
            '3M Ago', '6M Ago', '1Y Ago', '3Y Ago', '5Y Ago', 
            '10Y Ago', '15Y Ago', '20Y Ago', '30Y Ago', '50Y Ago', 'Custom']
        lo = Layout(width='80%', height='25px')
        w_dd_rel = widgets.Dropdown(
            options=lst_relative,
            value=self.settings.Relative_Opt,
            description='',
            disabled=False,
            layout=lo
        )

        lst_end = ['Previous Close'] + lst_relative
        w_dd_end = widgets.Dropdown(
            options=lst_end,
            value=self.settings.End_Opt,
            description='',
            disabled=False,
            layout=lo
        )

        def on_start_date_chg(chg):
            try:
                dt_tmp = pd.Timestamp(chg['owner'].value)
            except Exception as e:
                return

            if self.dt_end < dt_tmp:
                # error
                w_l_error.value = 'Start date must be earlier than end!'
                # w_d1.value = self.dt_start
                return
            w_l_error.value = ''

            self.dt_start = dt_tmp
            self.settings.Start_Date = self.dt_start
            self.notify()

            if self.b_start_auto:
                self.b_start_auto = False
                return
            
            w_dd_rel.value = 'Custom'
            
        def on_end_date_chg(chg):
            try:
                dt_tmp = pd.Timestamp(chg['owner'].value)
            except Exception as e:
                return

            if self.settings.Relative_Opt == 'Custom':
                if self.dt_start > dt_tmp:
                    # error
                    w_l_error.value = 'End date must be later than start!'
                    # w_d2.value = self.dt_end
                    return

            w_l_error.value = ''

            self.dt_end = dt_tmp
            self.settings.End_Date = self.dt_end
            self.notify()

            if self.settings.Relative_Opt != 'Custom':
                self.b_start_auto = True
                w_d1.value = offset(self.dt_end, self.settings.Relative_Opt)

            if self.b_end_auto:
                self.b_end_auto = False
                return

            w_dd_end.value = 'Custom'

        def on_rel_opt_chg(chg):
            self.settings.Relative_Opt = chg['owner'].value
            if chg['owner'].value == 'Custom':
                return

            self.b_start_auto = True
            w_d1.value = offset(self.dt_end, self.settings.Relative_Opt)

        def on_end_opt_chg(chg):
            self.settings.End_Opt = chg['owner'].value
            if chg['owner'].value == 'Custom':
                return

            self.b_end_auto = True
            w_d2.value = offset(self.dt_today, self.settings.End_Opt)

        w_d1.observe(on_start_date_chg, names='value')
        w_d2.observe(on_end_date_chg, names='value')
        w_dd_rel.observe(on_rel_opt_chg, names='value')
        w_dd_end.observe(on_end_opt_chg, names='value')
        w_box1 = widgets.VBox([w_l11, w_d1, w_l12, w_dd_rel])
        w_box2 = widgets.VBox([w_l21, w_d2, w_l22, w_dd_end])
        self.w_box  = widgets.VBox([widgets.HBox([w_box1, w_box2]), w_l_error])

    def run(self):
        display(self.w_box)

    @property
    def box(self):
        return self.w_box

    @property
    def start(self):
        return self.dt_start

    @property
    def end(self):
        return self.dt_end

    def notify(self):
        if self.fn_notify is not None:
            self.fn_notify(self.dt_start, self.dt_end)


class UIProgress(object):
    """docstring for UIProgress"""
    def __init__(self):
        super().__init__()
        self.__draw__()

    def __draw__(self):
        self.w_label = widgets.Label(value="")
        self.w_prog = widgets.IntProgress(
            value=7,
            min=0,
            max=10,
            step=1,
            description='',
            bar_style='', # 'success', 'info', 'warning', 'danger' or ''
            orientation='horizontal'
        )
        self.box = widgets.VBox([self.w_label, self.w_prog])

    def run(self):
        display(self.box)

    def reset(self, val, start, end):
        self.w_prog.value = val
        self.w_prog.min = start
        self.w_prog.max = end
        self.w_label.value = ""
        self.s_prefix = ""

    def info(self, msg):
        self.w_label.value = ' => '.join([self.s_prefix, msg])

    def increase(self, msg=None):
        self.w_prog.value += 1
        if msg is not None:
            self.w_label.value = msg
            self.s_prefix = msg

    def update(self, i, msg=None):
        self.w_prog.value = i
        if msg is not None:
            self.w_label.value = msg
            self.s_prefix = msg

    def close(self):
        self.box.close()

        