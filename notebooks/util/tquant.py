from enum import Enum
from datetime import date, timedelta, datetime
from dateutil.relativedelta import relativedelta
from dateutil.easter import *
from abc import ABC, abstractmethod
from calendar import monthrange
import tensorflow as tf
from tensorflow.python.framework import dtypes
import pandas as pd
import re
import math
import numpy as np

class Settings:
    """ 
    A class to represent general settings
    """
    evaluation_date = date.today()


# class DayCounter:
#     def __init__(self, name):
#         self.name = name

class Position(Enum):
    Long = 1
    Short = -1


class TimeUnit(Enum):
    """
    Enumeration of time-units.
    """ 
    Days = "Days"
    Weeks = "Weeks"
    Months = "Months"
    Years = "Years"

    def __str__(self):
        return self.value


class DayCounterConvention(Enum):
    Actual360 = "Actual360"
    Actual365 = "Actual365"
    Thirty360 = "Thirty360"
    Thirty360E = "Thirty360E"
    ActualActual = "ActualActual"

    def __str__(self):
        return self.value
    

class BusinessDayConvention(Enum):
    """
    Enumeration of business-day conventions.
    """ 
    Following = "Following"
    ModifiedFollowing = "Modified Following"
    HalfMonthModifiedFollowing = "Half-Month Modified Following"
    Preceding = "Preceding"
    ModifiedPreceding = "Modified Preceding"
    Unadjusted = "Unadjusted"
    Nearest = "Nearest"

    def __str__(self):
        return self.value
    

class CompoundingType(Enum):
    Compounded = "compounded"
    Simple = "simple"
    Continuous = "continuos"

    def __str__(self):
        return self.value
    
class Frequency(Enum):
    NoFrequency = -1
    Once = 0
    Annual = 1
    Semiannual = 2
    EveryFourthMonth = 3
    Quarterly = 4
    Bimonthly = 6
    Monthly = 12
    EveryFourthWeek = 13
    Biweekly = 26
    Weekly = 52
    Daily = 365
    OtherFrequency = 999

class InterpolationType(Enum):
    Linear = 'linear'
    Quadratic = "quadratic"
    Cubic = "cubic"

    def __str__(self):
        return self.value

class SwapType(Enum):
    """
    Enumeration of time-units.
    """ 
    Payer = "Payer"
    Receiver = "Receiver"

    def __str__(self):
        return self.value

class Calendar(ABC):
    ''' 
    The abstract class for calendar implementations.
    '''

    def advance(self,
                start_date: date,
                period: int,
                time_unit: TimeUnit,
                convention: BusinessDayConvention,
                end_of_month: bool = False):
        """
        Advance a given start date by a specified period using the given time unit and business day convention.

        Parameters:
        -------
            start_date (date): The starting date.
            period (int): The number of time units to advance the start date by.
            time_unit (TimeUnit): The unit of time to use for advancing the start date.
            convention (BusinessDayConvention): The business day convention to use for adjusting the dates.
            end_of_month (bool, optional): Whether to adjust to the end of the month if the original start date is at the end of the month. Defaults to False.

        Returns:
        -------
            date: The advanced date.
        """        
        if start_date is None:
            raise ValueError("null date")

        if period == 0:
            return self.adjust(start_date, convention)

        d1 = start_date

        if time_unit == TimeUnit.Days:
            while period > 0:
                d1 += timedelta(days=1)
                while self.is_holiday(d1):
                    d1 += timedelta(days=1)
                period -= 1

            while period < 0:
                d1 -= timedelta(days=1)
                while self.is_holiday(d1):
                    d1 -= timedelta(days=1)
                period += 1

        elif time_unit == TimeUnit.Weeks:
            d1 += relativedelta(weeks = period) 
            d1 = self.adjust(d1, convention)

        elif time_unit == TimeUnit.Months:
            d1 += relativedelta(months = period) 
            d1 = self.adjust(d1, convention)

            if end_of_month and self.is_end_of_month(start_date):
                return date(d1.year, d1.month, self.end_of_month(d1))
            
        else:  # Months or Years
            d1 += relativedelta(years = period) 
            d1 = self.adjust(d1, convention)
            if end_of_month and self.is_end_of_month(start_date):
                return date(d1.year, d1.month, self.end_of_month(d1))

        return d1

    def adjust(self,
               d: date,
               c: BusinessDayConvention):
        ''' 
        Adjust date based on the business day convention.

        Parameters:
        -------
            d (date): The date to adjust.
            c (BusinessDayConvention): The business day convention.

        Returns:
        -------
            date: The adjusted date.
        '''
        if d is None:
            raise ValueError("null date")

        if c == BusinessDayConvention.Unadjusted:
            return d

        d1 = d

        if c in [BusinessDayConvention.Following, BusinessDayConvention.ModifiedFollowing, BusinessDayConvention.HalfMonthModifiedFollowing]:
            while self.is_holiday(d1):
                d1 += timedelta(days=1)

            if c in [BusinessDayConvention.ModifiedFollowing, BusinessDayConvention.HalfMonthModifiedFollowing]:
                if d1.month != d.month:
                    return self.adjust(d, BusinessDayConvention.Preceding)

                if c == BusinessDayConvention.HalfMonthModifiedFollowing and d.day <= 15 and d1.day > 15:
                    return self.adjust(d, BusinessDayConvention.Preceding)

        elif c in [BusinessDayConvention.Preceding, BusinessDayConvention.ModifiedPreceding]:
            while self.is_holiday(d1):
                d1 -= timedelta(days=1)

            if c == BusinessDayConvention.ModifiedPreceding and d1.month != d.month:
                return self.adjust(d, BusinessDayConvention.Following)

        elif c == BusinessDayConvention.Nearest:
            d2 = d
            while self.is_holiday(d1) and self.is_holiday(d2):
                d1 += timedelta(days=1)
                d2 -= timedelta(days=1)

            if self.is_holiday(d1):
                return d2
            else:
                return d1

        else:
            raise ValueError("unknown business-day convention")

        return d1
    
    def is_end_of_month(self, d: date) -> bool:
        ''' 
        Check if a given date is the end of the month.

        Parameters:
        -------
            d (date): The date to check.

        Returns:
        -------
            bool: True if the given date is the end of the month, False otherwise.
        '''
        return d.month != self.adjust((d + timedelta(1)), 
                                        BusinessDayConvention.ModifiedFollowing).month
    
    def is_weekend(self, d: date) -> bool:
        ''' 
        Check if a given date is a weekend.

        Parameters:
        -------
            d (date): The date to check.

        Returns:
        -------
            bool: True if the given date is a weekend, False otherwise.
        '''
        return d.weekday() in [5,6]

    def end_of_month(self, d: date) -> int:
        ''' 
        Get the last day of the month for a given date.

        Parameters:
        -------
            d (date): The date for which to find the last day of the month.

        Returns:
        -------
            int: The last day of the month.
        '''
        _, last_day = monthrange(d.year, d.month)
        return last_day

    def is_holiday(self, d: date) -> bool:
        ''' 
        Check if a given date is a holiday.

        Parameters:
        -------
            d (date): The date to check.

        Returns:
        -------
            bool: True if the given date is a holiday, False otherwise.
        '''
        return not self.is_business_day(d)

    @abstractmethod
    def is_business_day(self, d: date):
        ''' 
        Check if a given date is a business day.

        Parameters:
        -------
            d (date): The date to check.

        Returns:
        -------
            bool: True if the given date is a business day, False otherwise.
        '''
        pass

class TARGET(Calendar):
    """
    A class to represent TARGET calendar
    """
    def __init__(self) -> None:
        pass 

    def is_business_day(self, d: date) -> bool:
        """
        Determine if a given date is a business day.
    
        Parameters:
        -------
        d : date
            The date to check.

        Returns:
        -------
        bool: 
            True if the given date is a business day, False otherwise.
        """
        ny = d.day == 1 and d.month == 1
        em = d == easter(d.year) + timedelta(1)
        gf = d == easter(d.year) - timedelta(2)
        ld = d.day == 1 and d.month == 5
        c = d.day == 25 and d.month == 12
        cg = d.day == 26 and d.month == 12

        if self.is_weekend(d) or ny or gf or em or ld or c or cg:
            return False
        else:
            return True

class DayCounter:

    def __init__(self,
                day_counter_convention: DayCounterConvention,
                include_last_day: bool = False
                ):
        self.day_counter_convention = day_counter_convention
        self.include_last_day = 1 if include_last_day else 0


    def year_days(self,
                  year: int):
        if year % 4 == 0:
            return 366
        else:
            return 365
    
    def year_fraction(self,
                  d1:date,
                  d2:date):
        if self.day_counter_convention == DayCounterConvention.Actual360:
            return self.day_count(d1,d2) / 360.0
        elif self.day_counter_convention == DayCounterConvention.Actual365:
            return self.day_count(d1,d2) / 365.0
        elif self.day_counter_convention == DayCounterConvention.Thirty360:
            return self.day_count(d1,d2) / 360.0
        elif self.day_counter_convention == DayCounterConvention.Thirty360E:
            return self.day_count(d1,d2) / 360.0
        elif self.day_counter_convention == DayCounterConvention.ActualActual:
            
            if d1.year == d2.year:
                return self.day_count(d1,d2)/self.year_days(d1.year)
            else:
                yearfraction = (date(d1.year,12,31) - d1).days/self.year_days(d1.year)
                date_year = d1.year
                while date_year < d2.year:
                    yearfraction = yearfraction + 1
                    date_year = date_year + 1
                yearfraction = (d2 - date(d2.year,1,1))/self.year_days(d2.year)
                return yearfraction

    def day_count(self,
                  d1:date,
                  d2:date):
        if self.day_counter_convention == DayCounterConvention.Actual360:
            return (d2-d1).days + self.include_last_day
        elif self.day_counter_convention == DayCounterConvention.Actual365 or self.day_counter_convention == DayCounterConvention.ActualActual:
            return (d2-d1).days
        elif self.day_counter_convention == DayCounterConvention.Thirty360:
            dd1 = d1.day
            dd2 = d2.day
            if dd1 == 31:
                dd1 = 30
            if dd2 == 31 and dd1 == 30:
                dd2 = 30
            return 360.0*(d2.year - d1.year) + 30.0*(d2.month - d1.month) + dd2 - dd1
        elif self.day_counter_convention == DayCounterConvention.Thirty360E:
            dd1 = d1.day
            dd2 = d2.day
            if dd1 == 31:
                dd1 = 30
            if dd2 == 31:
                dd2 = 30
            return 360.0*(d2.year - d1.year) + 30.0*(d2.month - d1.month) + dd2 - dd1
        

def set_period_unit(period_string):
    ''' 
    Trasforma il formato stringa-TimeUnit
    '''
    if period_string == 'Y':
        period_unit = TimeUnit.Years
    elif period_string == 'M':
        period_unit = TimeUnit.Months
    elif period_string == 'W':
        period_unit = TimeUnit.Weeks
    elif period_string == 'BD':
        period_unit = TimeUnit.Days
    return period_unit


class MarketDataLoader:
    def __init__(self,
                 evaluation_date: str) -> None:
        self.evaluation_date = evaluation_date
        self._load_curves() # load data from source
        self.eur_calendar = TARGET() # da fare mappaggio curva-calendario
        self._market_quotes = None
        self._market_discount = None

        self._ir_curve_data = None
        self._ir_curve_data_disc = None

        self._ir_vol_data = None
        self._ir_vol_data_disc = None

    def _load_curves(self):
        """
        Load market curves from the source.

        Returns:
            dict: Market data loaded from the specified source.
        """
        return self._load_local_data()
        
    def _load_local_data(self):
        """
        Load market data from a local source.

        Returns:
            dict: Market data loaded from a local source.
        """
        evaluation_date = self.evaluation_date
        y =  evaluation_date[:4]
        m = evaluation_date[4:6]
        d = evaluation_date[6:8]
        Settings.evaluation_date = date(int(y), int(m), int(d))
        date_ = y + '_' + m + '_' + d
        file_name_dfs = 'data/' +'Market_' + date_ + '_DFs.csv'
        file_name_quotes = 'data/' +'Market_' + date_ + '_Quotes.csv'

        self.raw_dfs = pd.read_csv(file_name_dfs,
                            skiprows= 2,
                            on_bad_lines='skip')

        self.raw_quotes = pd.read_csv(file_name_quotes,
                            skiprows= 2,
                            on_bad_lines='skip')
        

    def _load_database_data(self):
        """
        Load market data from a database.

        Returns:
            dict: Market data loaded from a database.
        """
        raise ValueError("Implement logic to load data from a database")
    
    @property
    def market_quotes(self):
        if self._market_quotes is None:
            self._filter_quotes()
        return self._market_quotes
    
    def _filter_quotes(self):
        ''' 
        splitta properties 
        '''
        quotes_properties = pd.DataFrame([self.raw_quotes['Property'][i].split(".") for i in range(self.raw_quotes.shape[0])] )
        market_quotes = pd.concat([quotes_properties, self.raw_quotes['Label']], axis=1)
        market_quotes.columns = ['type',
                                'name',
                                'curve_instrument',
                                'quote_type',
                                'bo1', # ??? 
                                'bo2',
                                'bo3',
                                'market_quote']
        self._market_quotes = market_quotes
    
    @property
    def market_discount(self):
        if self._market_discount is None:
            self._filter_discount()
        return self._market_discount
    
    def _filter_discount(self):
        quotes_properties = pd.DataFrame([self.raw_dfs['Property'][i].split(".") for i in range(self.raw_dfs.shape[0])] )
        market_discount = pd.concat([quotes_properties, self.raw_dfs['Label']], axis=1)
        market_discount.columns = ['type',
                                'name',
                                'curve_instrument',
                                'quote_type',
                                'bo1', # ??? 
                                'bo2',
                                'bo3',
                                'market_quote'] 
        self._market_discount = market_discount

    @property
    def ir_curve_data(self):
        if self._ir_curve_data is None:
            self._ir_quotes()
        return self._ir_curve_data
    
    @property
    def ir_vol_data(self):
        if self._ir_vol_data is None:
            self._ir_quotes()
        return self._ir_vol_data
    
    def _ir_quotes(self):
        ir_data = self.market_quotes[self.market_quotes['type'] == 'IR']
        ir_curve_data = ir_data[ir_data['quote_type'] == 'MID']

        self._ir_curve_data = ir_curve_data[['type', 'name', 'curve_instrument', 'quote_type', 'market_quote']]
        self._ir_vol_data = ir_data[ir_data['quote_type'] == 'SWPT']

    @property
    def ir_curve_data_disc(self):
        if self._ir_curve_data_disc is None:
            self._ir_discount()
        return self._ir_curve_data_disc
    
    @property
    def ir_vol_data_disc(self):
        if self._ir_vol_data_disc is None:
            self._ir_discount()
        return self._ir_vol_data_disc
    
    def _ir_discount(self):
        ir_data = self.market_discount[self.market_discount['type'] == 'IR']
        ir_curve_data = ir_data[ir_data['quote_type'] != 'SWPT']

        self._ir_curve_data_disc = ir_curve_data[['type', 'name', 'curve_instrument', 'quote_type', 'market_quote']]
        self._ir_vol_data_disc = ir_data[ir_data['quote_type'] == 'SWPT']

    @property
    def ir_eur_curve_1m(self):
        return self._ir_curves('EUR-EURIBOR-1M')
    @property
    def ir_eur_discount_1m(self):
        return self._ir_discounts('EUR-EURIBOR-1M')
    
    @property
    def ir_eur_curve_3m(self):
        return self._ir_curves('EUR-EURIBOR-3M')
    @property
    def ir_eur_discount_3m(self):
        return self._ir_discounts('EUR-EURIBOR-3M')
    
    @property
    def ir_eur_curve_6m(self):
        return self._ir_curves('EUR-EURIBOR-6M')
    @property
    def ir_eur_discount_6m(self):
        return self._ir_discounts('EUR-EURIBOR-6M')

    @property
    def ir_eur_curve_12m(self):
        return self._ir_curves('EUR-EURIBOR-12M')
    @property
    def ir_eur_discount_12m(self):
        return self._ir_discounts('EUR-EURIBOR-12M')
    
    @property
    def ir_eur_curve_estr(self):
        return self._ir_curves('EUR-ESTR-ON')
    @property
    def ir_eur_discount_estr(self):
        return self._ir_discounts('EUR-ESTR-ON')
    
    def _ir_curves(self, curve_name):
        ''' 
        Fa operazioni sulle stringhe per filtrare il df
        '''
        curve = self.ir_curve_data[self.ir_curve_data['name'] == curve_name]
        temp = re.compile("([0-9]+)([a-zA-Z]+)")
        df_curve = pd.DataFrame()
        for i in range( curve.shape[0]):
            split_temp = curve['curve_instrument'].iloc[i].split('-')
            if split_temp[0] == 'CASH' or split_temp[0] == 'SWAP':
                start_date = Settings.evaluation_date

                res = temp.match(split_temp[1]).groups()
                n_period = int(res[0])
                period_unit = set_period_unit(res[1]) 

                maturity = self.eur_calendar.advance(Settings.evaluation_date,
                                                    n_period, 
                                                    period_unit,
                                                    BusinessDayConvention.ModifiedFollowing 
                                                    )
                dt = (maturity - Settings.evaluation_date).days
                df_temp = pd.DataFrame( [split_temp[0], start_date, maturity, split_temp[1], dt, curve['market_quote'].iloc[i]] ).T
                df_curve = pd.concat([df_curve, df_temp], axis = 0)

            elif split_temp[0] == 'FRA':
                res_start = temp.match(split_temp[1]).groups()
                res_end = temp.match(split_temp[2]).groups()

                period_unit_start = set_period_unit(res_start[1]) 
                period_unit_end = set_period_unit(res_end[1]) 

                start_date = self.eur_calendar.advance(Settings.evaluation_date,
                                                    int(res_start[0]), 
                                                    period_unit_start,
                                                    BusinessDayConvention.ModifiedFollowing 
                                                    )
                maturity = self.eur_calendar.advance(start_date,
                                                    int(res_end[0]), 
                                                    period_unit_end,
                                                    BusinessDayConvention.ModifiedFollowing 
                                                    )
                dt = (maturity - Settings.evaluation_date).days
                df_temp = pd.DataFrame( [split_temp[0], start_date, maturity, curve['curve_instrument'].iloc[i], dt, curve['market_quote'].iloc[i]] ).T
                df_curve = pd.concat([df_curve, df_temp], axis = 0)


        df_curve.columns = ['type',
                            'start',
                            'maturity',
                            'tenor',
                            'daycount',
                            'quote']
        return df_curve

    def _ir_discounts(self, curve_name):
        ### 
        curve = self.ir_curve_data_disc[self.ir_curve_data_disc['name'] == curve_name].copy()
        curve = curve[curve['curve_instrument'] != 'PROJECTION']
        curve.reset_index(drop=True, inplace=True)
        curve_dates_str = [curve['curve_instrument'].iloc[i].replace('DF-','') for i in range(curve.shape[0])]
        curve_dates = pd.DataFrame([datetime.strptime(str_temp, '%d-%b-%Y').date() for str_temp in curve_dates_str],
                                columns = ['maturity_date'])
        dt = pd.DataFrame([i.days for i in (curve_dates.loc[:,'maturity_date'] - Settings.evaluation_date) ],
                        columns = ['daycount'])
        curve = pd.concat([curve, curve_dates, dt], axis = 1)
        return curve
    
class RateCurve:
    def __init__(self, pillars, rates):
        ''' 
        Classe dummy per le curve
        '''
        self.pillars = pillars # list
        self.rates = [tf.Variable(r, dtype=dtypes.float64) for r in rates]     # tensor list

    def discount(self, term):
        if term <= self.pillars[0]:
            nrt = -term * self.rates[0]
            df = tf.exp(nrt)
            return df
        if term >= self.pillars[-1]:
            nrt = -term * self.rates[-1]
            df = tf.exp(nrt)
            return df
        for i in range(0, len(self.pillars) - 1):
            if term < self.pillars[i + 1]:
                dtr = 1 / (self.pillars[i + 1] - self.pillars[i])
                w1 = (self.pillars[i + 1] - term) * dtr
                w2 = (term - self.pillars[i]) * dtr
                r1 = w1 * self.rates[i]
                r2 = w2 * self.rates[i + 1]
                rm = r1 + r2
                nrt = -term * rm
                df = tf.exp(nrt)
                return df
            
    def inst_fwd(self, t: float):
        dt = 0.01    
        expr = - (tf.math.log(self.discount(t+dt))- tf.math.log(self.discount(t-dt)))/(2*dt)
        return expr
    
    def forward_rate(self,
                    d1,
                    d2,
                    daycounter,
                    evaluation_date):
        ''' 
        Calcola il tasso forward.
        '''
        tau = daycounter.year_fraction(d1, d2)
        df1 = self.discount(daycounter.year_fraction(evaluation_date, d1))  
        df2 = self.discount(daycounter.year_fraction(evaluation_date, d2)) 
        return (df1 / df2 -1) / tau

class Index(ABC):
    """
    Abstract class representing an index.

    Attributes:
    -----------
        _name: str
            The name of the index.
        _fixing_calendar Calendar
            The calendar used for determining fixing dates.
        _fixing_time_series: dict
            A dictionary containing fixing time series data.

    """
    @abstractmethod
    def __init__(self,
                 name: str,
                 fixing_calendar: Calendar,
                 fixing_time_series: dict 
                 ) -> None:
        self._name = name
        self._fixing_calendar = fixing_calendar
        self._fixing_time_series = fixing_time_series

    @property
    def name(self) -> str:
        """
        Get the name of the index.

        Returns:
        -----------
            str: The name of the index.

        """
        return self._name
    
    @property
    def fixing_time_series(self)-> dict:
        """
        Get the fixing time series data.

        Returns:
        -----------
            dict: A dictionary containing fixing time series data.

        """
        return self._fixing_time_series  
    
    @fixing_time_series.setter
    def fixing_time_series(self, input_fixings)-> None:
        """
        Set the fixing time series data.

        Parameters:
        -----------
            input_fixings: The fixing time series data to be set.
        """
        self._fixing_time_series = input_fixings
 
    @property
    def fixing_calendar(self) -> Calendar:
        """
        Get the fixing calendar of the index.

        Returns:
        -----------
            str: The fixing calendar of the index.

        """
        return self._fixing_calendar

    def is_valid_fixing_date(self, date: date) -> bool:
        """
        Check if the given date is a valid fixing date.

        Parameters:
        -----------
            date: date
                The date to be checked.

        Returns:
        -----------
            bool: True if the date is a valid fixing date, False otherwise.

        """
        # TODO da implementare la funzione in modo che valuti se la data in input sia valida dato un calendario
        return True 
    
    def add_fixing(self, date: date, value: float)-> None:
        """
        Add a fixing value for a specific date to the fixing time series data.

        Parameters:
        -----------
            date: date
                The date of the fixing.
            value: float
                The fixing value.
        """
        fixing_point = {
            self.name: {
                date: value
            }
        }
        if self.fixing_time_series is None:
            # create the dict
            self.fixing_time_series = fixing_point
        else:
            # write into it
            self.fixing_time_series[self.name][date] = value 

    def past_fixing(self, date: date)-> bool:
        """
        Get the past fixing value for a specific date.

        Parameters:
        -----------
            date: date
                The date for which past fixing is requested.

        Returns:
        -----------
            bool: The past fixing value for the given date.

        Raises:
            ValueError: If the given date is not a valid fixing date.

        """
        past_fixings = self.fixing_time_series
        if self.is_valid_fixing_date(date):
            return past_fixings[self.name][date]
        else:
            raise ValueError("Not a valid fixing date!")

    def fixing(self,
               date
               )-> float:
        """
        Get the fixing value for a specific date.

        Parameters:
        -----------
            date: date
                The date for which fixing is requested.

        Returns:
        -----------
            float: The fixing value for the given date.

        Raises:
        -----------
            ValueError: If the given date is not a valid date.

        """
        if not self.is_valid_fixing_date:
            raise ValueError("Not a valid date")
        
        if date >= Settings.evaluation_date:
            # time series empty, try to forecast the fixing 
            raise ValueError("Fixing are only available for historical dates.")
            # return self.forecast_fixing(date, term_structure)
        
        elif date < Settings.evaluation_date:
            if self.fixing_time_series is None:
                raise ValueError(f"Missing {self.name} fixing for {date}") 
            
            if self.name in list(self.fixing_time_series.keys()):
                if date in list(self.fixing_time_series[self.name].keys()):
                    # return historical fixing for index/date
                    return self.past_fixing(date)
                else:
                    raise ValueError(f"{self.name} fixing time series is not complete, missing {date}")
            else:
                raise ValueError(f"Missing {self.name} fixing for {date}")
          

class IborIndex(Index):
    """
    Represents an Interbank Offered Rate (IBOR) index.

    Attributes:
    -----------
        name: str
            The name of the index.
        fixing_calendar: Calendar
            The calendar used for determining fixing dates.
        tenor: int
            The tenor of the index.
        time_unit: TimeUnit
            The time unit for the tenor.
        fixing_days: int, optional
            The number of days for fixing. Defaults to None.
        time_series: dict, optional
            A dictionary containing time series data. Defaults to None.

    Note:
    -----------
        Inherits from Index abstract class.

    """
    def __init__(self,
                 name: str,
                 fixing_calendar: Calendar,
                 tenor: int,
                 time_unit: TimeUnit,
                 fixing_days: int = None,
                 time_series: dict = None) -> None:
        super().__init__(name,
                         fixing_calendar,
                         time_series)      
        self._fixing_days = fixing_days
        self._tenor = tenor 
        self._time_unit = time_unit        
    
    @property
    def fixing_days(self) -> int:
        """
        Get the number of fixing days for the index.

        Returns:
        -----------
            int: The number of fixing days if set, otherwise 0.

        """
        if self._fixing_days == None:
            return 0
        else:
            return self._fixing_days
    
    def fixing_maturity(self, fixing_date: date) -> date:
        """
        Calculate the fixing maturity date based on the fixing date and index conventions.

        Parameters:
        -----------
            fixing_date: date
                The fixing date.

        Returns:
        -----------
            date: The maturity date for the fixing helper.

        """
        return self.fixing_calendar.advance(fixing_date,
                                            self._tenor, 
                                            self._time_unit,
                                            BusinessDayConvention.ModifiedFollowing 
                                            )

    def fixing_date(self, value_date: date) -> date:
        return self.fixing_calendar.advance(value_date, self.fixing_days, TimeUnit.Days, BusinessDayConvention.ModifiedFollowing)
   

class Trade(ABC):
    
    @abstractmethod
    def __init__(self) -> None:
        self._trade_id = None
        self._currency = None
        self._product_type = None
        self._counterparty = None
        self._netting_set = None
    
    @property
    def trade_id(self):
        if self._trade_id is not None:
            return self._trade_id
        else:
            raise ValueError("Trade ID not assigned")
    @trade_id.setter
    def trade_id(self, value):
        self._trade_id = value
        
    @property
    def currency(self):
        if self._currency is not None:
            return self._currency
        else:
            raise ValueError("Currency not assigned")
    @currency.setter
    def currency(self, value):
        self._currency = value
        
    @property
    def product_type(self):
        if self._product_type is not None:
            return self._product_type
        else:
            raise ValueError("not assigned")
    @product_type.setter
    def product_type(self, value):
        self._product_type = value
        
    @property
    def counterparty(self):
        if self._counterparty is not None:
            return self._counterparty
        else:
            raise ValueError("not assigned")
    @counterparty.setter
    def counterparty(self, value):
        self._counterparty = value
        
    @property
    def netting_set(self):
        if self._netting_set is not None:
            return self._netting_set
        else:
            raise ValueError("not assigned")
    @netting_set.setter
    def netting_set(self, value):
        self._netting_set = value


class InterestRate:
    
    def __init__(self,
                 r: float,
                 daycounter: DayCounter,
                 compounding: CompoundingType, 
                 frequency: Frequency) -> None:
        self._r = r
        self._daycounter = daycounter
        self._compounding = compounding
        self._frequency = frequency.value

    @property
    def rate(self):
        return self._r

    @property
    def daycounter(self):
        return self._daycounter
    
    @property
    def compounding(self):
        return self._compounding

    @property
    def frequency(self):
        return self._frequency
    
    def discount_factor(self, t:float) -> float:
        return 1/self.compound_factor(t)
    
    def compound_factor(self, *args):
        ''' 
        
        Da implementare il daycounter.yearfraction riga 35
        '''
        if len(args) == 1:
            t = args[0]
        elif len(args) == 4:
            d1, d2, refStart, refEnd = args
            # tau_i = (d2 - d1) 
            # t = tau_i.days/365 #dc_.yearFraction(d1, d2, refStart, refEnd)
            t = self._daycounter.year_fraction(d1, d2)
            return self.compound_factor(t)
        else:
            raise ValueError("Invalid number of arguments")
        
        if t < 0.0:
            raise ValueError(f"Negative time ({t}) not allowed")
        if self._r is None:
            raise ValueError("Null interest rate")

        if self.compounding == CompoundingType.Simple:
            return 1.0 + self._r * t
        elif self.compounding == CompoundingType.Compounded:
            return math.pow(1.0 + self._r / self.frequency, self.frequency * t)
        elif self.compounding == CompoundingType.Continuous:
            return math.exp(self._r * t)
        else:
            raise ValueError("Unknown compounding convention")
        
    
    @staticmethod
    def implied_rate(compound, daycounter, comp, freq, t):
        ''' 
        Returns the InterestRate object given a compound factor
        '''
        if compound <= 0.0:
            raise ValueError("Positive compound factor required")

        r = None
        if compound == 1.0:
            if t < 0.0:
                raise ValueError(f"Non-negative time ({t}) required")
            r = 0.0
        else:
            if t <= 0.0:
                raise ValueError(f"Positive time ({t}) required")
            if comp == CompoundingType.Simple:
                r = (compound - 1.0) / t
            elif comp == CompoundingType.Compounded:
                r = (math.pow(compound, 1.0 / (freq * t)) - 1.0) * freq
            elif comp == CompoundingType.Continuous:
                r = math.log(compound) / t
            else:
                raise ValueError(f"Unknown compounding convention ({int(comp)})")
        return InterestRate(r, daycounter, comp, freq)
    

    def __str__(self):
        if self._r is None:
            return "null interest rate"
        result = f"{self.rate:.6f}"
        if self.compounding == CompoundingType.Simple:
            result += " simple compounding"
        elif self.compounding == CompoundingType.Compounded:
            result += f" {self.frequency} compounding"
        elif self.compounding == CompoundingType.Continuous:
            result += " continuous compounding"
        else:
            raise ValueError(f"Unknown compounding convention ({int(self.compounding)})")
        return result


class CashFlow(ABC):
    ''' 
    The abstract class for CashFlow implementations.
    '''
    @abstractmethod
    def date(self) -> date:
        ''' 
        Payment date of the cashflow
        '''
        pass
        
    @abstractmethod
    def amount(self) -> float:
        ''' 
        Future amount (not discounted) of the cashflow.
        '''
        pass

    def has_occurred(self,
                     ref_date: date) -> bool:
        ''' 
        Check if a given date cashflow is in the past.

        Parameters:
        -------
            ref_date: date
                The date to check.

        Returns:
        -------
            bool: True if the given cashflow has occurred, False otherwise.
        '''
        cf = self.date
        if cf <= ref_date:
            return True
        if cf > ref_date:
            return False

class Coupon(CashFlow, ABC):
# TODO rivedere struttura della classe
    @abstractmethod
    def __init__(self,
                 payment_date: date,
                 nominal,
                 daycounter: DayCounter,
                 accrual_start_date: date,
                 accrual_end_date: date,
                 ref_period_start: date,
                 ref_period_end: date):
        
        self._payment_date = payment_date
        self._nominal = nominal
        self._daycounter = daycounter
        self._accrual_start_date = accrual_start_date
        self._accrual_end_date = accrual_end_date
        self._ref_period_start = ref_period_start
        self._ref_period_end = ref_period_end
        # self._ex_coupon_date = ex_coupon_date
    
    @abstractmethod
    def rate(self):
        return 
    
    @property
    @abstractmethod
    def accrued_amount(self):
        #TODO da capire come gestire l'accrued amount
        return
    
    @property
    def date(self):
        ''' 
        payment date
        '''
        return self._payment_date

    @property
    def nominal(self):
        ''' 
        nominal
        '''
        return self._nominal
    
    @property
    def daycounter(self):
        return self._daycounter
    
    @property
    def accrual_start_date(self):
        ''' 
        period accrual start
        '''
        return self._accrual_start_date

    @property
    def accrual_end_date(self):
        ''' 
        period accrual end
        '''
        return self._accrual_end_date

    @property
    def ref_period_start(self):
        return self._ref_period_start

    @property
    def ref_period_end(self):
        return self._ref_period_end

    @property
    def accrual_period(self):
        #TODO da validare -- vedi riga 44 coupon.cpp
        return self._daycounter.year_fraction(self.accrual_start_date,
                                                 self.accrual_end_date
                                            )  
    
    @property
    def accrual_days(self):
        #TODO da validare -- da implementare dayCount(accrualstart, accrualend) vedi r52 coupon.cpp
        return self._daycounter.day_count(self.accrual_start_date,
                                            self.accrual_end_date
                                            )
        
    @property
    def accrued_period(self, d: date):
        #TODO da validare -- riga 57 coupon.cpp
        return self._daycounter.year_fraction(self.accrual_start_date,
                                            min(d, self.accrual_end_date)
                                            ) 

    @property
    def accrued_days(self, d: date):
        #TODO da validare -- riga 71 coupon.cpp
        return self._daycounter.day_count(self.accrual_start_date,
                                            min(d, self.accrual_end_date)
                                            )

class FixedCoupon(Coupon):
    ''' 
    Concrete Fixed Coupon object. 
    
    '''
    def __init__(self,
                 payment_date: date,
                 nominal: float,
                 accrual_start_date: date,
                 accrual_end_date: date,
                 ref_period_start: date,
                 ref_period_end: date,
                 r: float,
                 daycounter: DayCounter
                 ):
        super().__init__(payment_date, nominal, daycounter, accrual_start_date, accrual_end_date, ref_period_start, ref_period_end)
        self._rate = InterestRate(r, daycounter, CompoundingType.Simple, Frequency.Annual)
        self._daycounter = daycounter

        
    @property
    def rate(self):
        return self._rate
    
    @property
    def day_counter(self):
        return self._daycounter
    
    def display(self):
        coupon_display = pd.DataFrame([self.ref_period_start,
                                        self.ref_period_end,
                                        self.date,
                                        self._nominal,
                                        self.accrual_period,
                                        self._daycounter.day_counter_convention.name,
                                        self._rate.rate,
                                        self.amount
                                        ]).T

        coupon_display.columns = ['start_period',
                                'end_period',
                                'payment_date',
                                'notional',
                                'accrual',
                                'day_counter',
                                'rate',
                                'amount'
                                ]
        return coupon_display
    
    @property
    def amount(self)-> float:
        self._amount = self.nominal * (self._rate.compound_factor(self.accrual_start_date,
                                                                    self.accrual_end_date,
                                                                    self.ref_period_start,
                                                                    self.ref_period_end
                                                                    ) - 1)
        return self._amount
    
    @property
    def accrual_period(self):
        return self._daycounter.year_fraction(self.accrual_start_date, self.accrual_end_date)
        
    
    def accrued_amount(self, d: date): 
        if d <= self.accrual_start_date or d > self._payment_date:
            return 0
        else:
            return self.nominal * (self._rate.compound_factor(self.accrual_start_date,
                                                            min(d, self.accrual_end_date),
                                                            self.ref_period_start,
                                                            self.ref_period_end
                                                            ) - 1)


class FloatingCoupon(Coupon):
    ''' 
    Concrete Floating Coupon object. 
    '''
    def __init__(self,
                 payment_date: date,
                 nominal: float,
                 accrual_start_date: date,
                 accrual_end_date: date,
                 index: IborIndex, 
                 gearing: float, # multiplicative coefficient of the index fixing
                 spread: float, # fixed spread
                 ref_period_start: date,
                 ref_period_end: date,
                 daycounter: DayCounter,
                 is_in_arrears: bool = False, #TODO da implementare convexity adjustments
                 fixing_days: int = None,
                 ): 
        super().__init__(payment_date, nominal, daycounter, accrual_start_date, accrual_end_date, ref_period_start, ref_period_end)
        self._day_counter = daycounter
        self._fixing_days = fixing_days 
        self._index = index
        self._gearing = gearing
        self._spread = spread
        self._is_in_arrears = is_in_arrears

    @property
    def day_counter(self):
        return self._day_counter
    
    @property
    def fixing_days(self):
        if self._fixing_days is None:
            if self.index is not None:
                return self.index.fixing_days
            else:
                return 0
    
    @property
    def index(self):
        return self._index
            
    @property
    def is_in_arrears(self):
        return self._is_in_arrears
    
    @property
    def fixing_date(self):
        if self.is_in_arrears:
            ref_date = self.accrual_end_date 
        else:
            ref_date = self.accrual_start_date
        return ref_date + timedelta(self.fixing_days)
        
    @property
    def accrual_period(self): 
        return self._daycounter.year_fraction(self.accrual_start_date, self.accrual_end_date)

    
    def display(self):

        coupon_display = pd.DataFrame([self.ref_period_start,
                                        self.ref_period_end,
                                        self.date,
                                        self._nominal,
                                        self.fixing_date,
                                        self._fixing_days,
                                        self._index.name,
                                        self.accrual_period,
                                        self.is_in_arrears,
                                        self._gearing,
                                        self._spread,
                                        self._daycounter.day_counter_convention.name
                                        ]).T

        coupon_display.columns = ['start_period',
                                'end_period',
                                'payment_date',
                                'notional',
                                'fixing_date',
                                'fixing_days',
                                'index',
                                'accrual',
                                'in_arrears',
                                'gearing',
                                'spread',
                                'day_counter'
                                ]
        return coupon_display
    
    def amount(self, coupon_pricer)-> float: #TODO creare il pricer
        ''' 
        Da testare funzione _rate.compoundFactor (versione overloaded su file interestrate.hpp)
        '''
        a = self.nominal * (self._gearing * self.rate(coupon_pricer) + self._spread) * self.accrual_period
        return a
    
    def rate(self, coupon_pricer): #TODO
        self._rate = coupon_pricer.forecasted_rate
        return coupon_pricer.forecasted_rate
       
    def accrued_amount(self, d: date):
        if d <= self.accrual_start_date or d > self._payment_date:
            return 0
        else:
            return self.nominal * (self._rate.compound_factor(self.accrual_start_date,
                                                            min(d, self.accrual_end_date),
                                                            self.ref_period_start,
                                                            self.ref_period_end
                                                            ) - 1)
        
    
class FixedRateLeg:
    def __init__(self,
                 schedule: list[date],
                 notionals: list[float],
                 coupon_rates: list[float], 
                 daycounter: DayCounter,
                 compounding: CompoundingType = CompoundingType.Simple,
                 frequency: Frequency = Frequency.Annual) -> None:
        self._schedule = schedule
        self._notionals = notionals
        self._rates = coupon_rates
        self._daycounter = daycounter
        self._compounding = compounding
        self._frequency = frequency
  
    @property
    def schedule(self):
        return self._schedule
    
    @property
    def notionals(self):
        return self._notionals
    
    @property
    def frequency(self):
        return self._frequency
    
    @property
    def compounding(self):
        return self._compounding

    @property
    def coupon_rates(self):
        return [InterestRate(r, self._daycounter, self._compounding, self._frequency) for r in self._rates]

    @property
    def payment_adjustment(self):
        return self._payment_adjustment 
    
    @payment_adjustment.setter
    def payment_adjustment(self, payment_adjustment: float):
        self._payment_adjustment = payment_adjustment


    def leg_flows(self):
        ''' 
        Define the leg as a list of FixedCoupon objects
        TBD: definire bene tutti gli accrual 
        '''
        leg = []
        for i in range(1, len(self._schedule)):
            period_start_date = self._schedule[i-1]
            payment_date = self.schedule[i]
            nom = self._notionals[i-1]
            r = self._rates[i-1]
            leg.append(FixedCoupon(payment_date,
                                    nom,
                                    period_start_date,
                                    payment_date,
                                    period_start_date,
                                    payment_date,
                                    r,
                                    self._daycounter)
                            )
        return leg
    
    def display_flows(self):
        flows = self.leg_flows()
        leg_display = pd.DataFrame()
        for i in range(len(flows)):
            coupon_flow = flows[i].display()
            leg_display = pd.concat([leg_display, coupon_flow], axis = 0)
        return leg_display
    
class FloatingRateLeg:
    def __init__(self,
                 schedule: list[date],
                 notionals: list[float],
                 gearings: list[float],
                 spreads: list[float],
                 index: IborIndex,
                 daycounter: DayCounter
                 ) -> None:
        self._schedule = schedule
        self._notionals = notionals
        self._gearings = gearings
        self._spreads = spreads 
        self._index = index
        self._daycounter = daycounter
  
    @property
    def schedule(self):
        return self._schedule
    
    @property
    def notionals(self):
        return self._notionals
    
    @property
    def gearings(self):
        return self._gearings
    
    @property
    def spreads(self):
        return self._spreads
    
    @property
    def index(self):
        return self._index
    
    @property
    def daycounter(self):
        return self._daycounter
    
    def leg_flows(self):
        ''' 
        Define the leg as a list of FixedCoupon objects
        TBD: definire bene tutti gli accrual 
        '''
        leg = []
        for i in range(1, len(self._schedule)):
            period_start_date = self._schedule[i-1]
            payment_date = self.schedule[i]
            nom = self._notionals[i-1]

            leg.append(FloatingCoupon(payment_date,
                                    nom,
                                    period_start_date,
                                    payment_date,
                                    self._index,
                                    self._gearings[i-1],
                                    self._spreads[i-1],
                                    period_start_date,
                                    payment_date,
                                    self._daycounter
                                    )
                            )
        return leg
    
    def display_flows(self):
        flows = self.leg_flows()
        leg_display = pd.DataFrame()
        for i in range(len(flows)):
            coupon_flow = flows[i].display()
            leg_display = pd.concat([leg_display, coupon_flow], axis = 0)
        return leg_display

        


class InterestRateSwap(Trade):
    def __init__(self, 
                float_schedule: list[date],
                fix_schedule: list[date],
                float_notionals: list[float],
                fix_notionals: list[float],
                gearings: list[float],
                spreads: list[float],
                index: Index,
                fixed_coupons,
                fixed_daycounter: DayCounterConvention,
                floating_daycounter: DayCounterConvention
                ) -> None:
        super().__init__()
        self.fixed_leg = FixedRateLeg(fix_schedule, fix_notionals, fixed_coupons, fixed_daycounter)
        self.floating_leg = FloatingRateLeg(float_schedule, float_notionals, gearings, spreads, index, floating_daycounter)



class Pricer(ABC):

    @abstractmethod
    def price(self):
        return 

    @abstractmethod
    def price_aad(self):
        return
    
class FloatingCouponDiscounting(Pricer):

    def __init__(self,
                 coupon: FloatingCoupon,
                 convexity_adjustment: bool) -> None:
        self._coupon = coupon
        self._convexity_adj = convexity_adjustment

    def floating_rate(self, fixing_date, term_structure, evaluation_date):
        if self._convexity_adj:
            raise ValueError("Convexity Adjustment da implementare") #TODO
        else:
            if fixing_date >= Settings.evaluation_date: # forecast
                d2 = self._coupon.index.fixing_maturity(fixing_date)
                return term_structure.forward_rate(fixing_date, d2, self._coupon.day_counter, evaluation_date) 
            else: # historical
                return self._coupon.index.fixing(self._coupon.fixing_date)

    def amount(self, term_structure, evaluation_date)-> float: 
        ''' 
        cash flow futuro non scontato
        '''
        a = self._coupon.nominal * (self._coupon._gearing*self.floating_rate(self._coupon.fixing_date,term_structure,evaluation_date) + self._coupon._spread) * self._coupon.accrual_period
        return a

    def price(self, discount_curve: RateCurve, estimation_curve, evaluation_date: date):
        if not self._coupon.has_occurred(evaluation_date):
            tau = self._coupon.day_counter.year_fraction(evaluation_date, self._coupon._payment_date)
            return self.amount(estimation_curve, evaluation_date) * discount_curve.discount(tau)
        else:
            return 0
           
    def price_aad(self, discount_curve: RateCurve, estimation_curve, evaluation_date: date):
        with tf.GradientTape() as tape:
            npv = self.price(discount_curve, estimation_curve, evaluation_date)
        return npv, tape


class FloatingLegDiscounting(Pricer):

    def __init__(self,
                 leg: FloatingRateLeg) -> None:
        self._leg = leg

    def price(self, discount_curve, estimation_curve, evaluation_date: date, coupon_pricer: Pricer):
        if len(self._leg.leg_flows()) == 0:
            return 0
        npv = 0
        for i in range(0, len(self._leg.leg_flows())):
            cf = self._leg.leg_flows()[i]
            if not cf.has_occurred(evaluation_date):
                pricer = coupon_pricer(cf, False)
                npv += pricer.price(discount_curve, estimation_curve, evaluation_date)
        return npv

    def price_aad(self, discount_curve: RateCurve, estimation_curve, evaluation_date: date, coupon_pricer: Pricer):
        with tf.GradientTape() as tape:
            npv = self.price(discount_curve, estimation_curve, evaluation_date, coupon_pricer)
        return npv, tape
    
class FixedCouponDiscounting(Pricer):

    def __init__(self,
                 coupon: FixedCoupon) -> None:
        self._coupon = coupon

    def price(self, discount_curve: RateCurve, evaluation_date: date):
        if not self._coupon.has_occurred(evaluation_date):
            tau = self._coupon.day_counter.year_fraction(evaluation_date, self._coupon._payment_date)
            return self._coupon.amount * discount_curve.discount(tau)
        else:
            return 0
           
    def price_aad(self, discount_curve: RateCurve, evaluation_date: date):
        with tf.GradientTape() as tape:
            npv = self.price(discount_curve, evaluation_date)
        return npv, tape
    

class FixedLegDiscounting(Pricer):

    def __init__(self,
                 leg: FixedRateLeg) -> None:
        self._leg = leg

    def price(self, discount_curve, evaluation_date: date, coupon_pricer: Pricer):
        if len(self._leg.leg_flows()) == 0:
            return 0
        npv = 0
        for i in range(0, len(self._leg.leg_flows())):
            cf = self._leg.leg_flows()[i]
            if not cf.has_occurred(evaluation_date):
                pricer = coupon_pricer(cf)
                npv += pricer.price(discount_curve, evaluation_date)
        return npv

    def price_aad(self, discount_curve: RateCurve, evaluation_date: date, coupon_pricer: Pricer):
        with tf.GradientTape() as tape:
            npv = self.price(discount_curve, evaluation_date, coupon_pricer)
        return npv, tape

class SwapAnalyticEngine(Pricer):

    def __init__(self, swap: InterestRateSwap) -> None:
        self.swap = swap
        self.floating_leg_pricer = FloatingLegDiscounting(swap.floating_leg)
        self.fixed_leg_pricer = FixedLegDiscounting(swap.fixed_leg)

    def price(self, discount_curve, estimation_curve, evaluation_date: date):
        npv_float = self.floating_leg_pricer.price(discount_curve, estimation_curve, evaluation_date, FloatingCouponDiscounting)
        npv_fixed = self.fixed_leg_pricer.price(discount_curve, evaluation_date, FixedCouponDiscounting)
        return npv_fixed + npv_float
    
    def price_aad(self, discount_curve, estimation_curve, evaluation_date: date):
        with tf.GradientTape() as tape:
            npv = self.price(discount_curve, estimation_curve, evaluation_date)
        return npv, tape

    def implied_rate(self):
        pass



class HullWhiteProcess:
    def __init__(self, mean_rev: float, sigma: float, market_curve) -> None:
        self.sigma = tf.Variable(mean_rev, dtype=dtypes.float64)
        self.mean_rev = tf.Variable(sigma, dtype=dtypes.float64)
        self.market_curve = market_curve
        
    
    def alpha(self, t: float) -> float:
        """ 
        This function returns the alpha 
        time-dependent parameter.
        α(t) = f(0, t) + 0.5(σ(1-exp(-kt))/k)^2
        
        Parameters:
        t : reference time in years.
        
        Returns:
        α(t) : deterministic parameter to recover term-rates.
        """
        f = self.market_curve.inst_fwd(t)
        return f + self.sigma**2 / (2 * self.mean_rev**2) * (1 - math.exp(- self.mean_rev*t))**2
        

    def conditional_moments(self, s: float, t: float, r_s: tf.Tensor) -> tuple[tf.Tensor, float]:
        """ 
        This function returns the conditional mean
        and conditional variance of the short rate process 
        given the known value
        at time s <= t.
        E{r(t)|r(s)} = (r(s) - α(s))*exp(-k(t-s)) + α(t)
        Var{r(t)|r(s)} = σ^2[1 - exp(-2k(t-s))]/(2k)
        
        Parameters:
        s : information stopping time in years.
        t : reference time in years.
        r_s : short rate known at time s.
        
        Returns:
        E{r(t)|r(s)} : conditional mean
        Var{r(t)|r(s)} : conditional variance
        """
        dt = t-s
        a_t = self.alpha(t)
        a_s = self.alpha(s)
        decay = math.exp(- self.kappa * dt)
        E_rt = r_s * decay + a_t - a_s * decay
        Var_rt = self.sigma**2 / (2*self.kappa) * (1 - math.exp(-2*self.kappa*dt))
        return E_rt, Var_rt
    
    def A_B(self, S: float, T: float) -> tuple[float, float]:
        """ 
        This function returns the time dependent parameters
        of the ZCB, where S <= T.
        B(S, T) = (1 - exp(-k(T-S)))/k
        A(S, T) = P(0,T)/P(0,S) exp(B(S,T)f(0,S) - 
                    σ^2(exp(-kT)-exp(-kS))^2(exp(2kS)-1)/(4k^3))
        
        Parameters :
        S : future reference time of the ZCB in years.
        T : future reference maturity of the ZCB years.
        
        Returns : 
        A(S, T) : scale factor of the ZCB
        B(S, T) : exponential factor of the ZCB
        """
        f0S = self.market_curve.inst_fwd(S) 
        P0T = self.market_curve.discount(T)
        P0S = self.market_curve.discount(S)

        B = 1 - tf.exp(-self.mean_rev*(T - S))
        B /= self.mean_rev
        
        exponent = self.sigma*(tf.exp(-self.mean_rev*T) - tf.exp(-self.mean_rev*S))
        exponent *= exponent
        exponent *= tf.exp(2*self.mean_rev*S) - 1
        exponent /= -4*(self.mean_rev**3)
        exponent += B*f0S
        A = tf.exp(exponent)*P0T/P0S
        return A, B
    
    def zero_bond(self, S: float, T: float, rs: float) -> float:
        """ 
        This function returns the price of a ZCB
        P(S, T) at future reference time S and maturity T 
        with S <= T.
        
        Parameters :
        S : future reference time of the ZCB in years.
        T : future reference maturity of the ZCB years. 
        
        Returns :
        P(S, T) : ZCB price with maturity T at future date S.
        """
        A, B = self.A_B(S, T)
        return A*tf.exp(-B*rs)
    
class GaussianRateKernel1D:
    def __init__(self, process: HullWhiteProcess) -> None:
        self.process = process

    def rate_tensor(self, 
                    nPaths: int,
                    last_grid_time: float, 
                    time_steps: float,
                    time_grid: np.array = None) -> tuple[tf.Tensor, np.array]:
        """ 
        This function returns a path drawn from
        the Gaussian distribution of the conditional short rate.
        
        Parameters:
        last_grid_time : time span in years.
        time_steps : number of steps in the discretized path.
        
        Returns:
        times : array containing the times points in years.
        rt_path : array containing the short rate points.
        """
        if time_grid is None:
            time_grid = np.linspace(start=0.0, stop=last_grid_time, num=time_steps, retstep=False)
        W = tf.random.normal(shape=(nPaths, len(time_grid)), mean=0, stddev=1, dtype= tf.float64)
        pillars_dfs = self.process.market_curve.pillars

        rates = []
        zb_tensor = []
        rates.append(tf.fill((nPaths,),
                             value= self.process.market_curve.inst_fwd(0)))

        for i in range(1, len(time_grid)):

            s = time_grid[i-1]
            t = time_grid[i]
            dt = t-s
            cond_var = self.process.sigma**2 / (2*self.process.mean_rev) * (1 - math.exp(-2*self.process.mean_rev*dt))
            std_dev = math.sqrt(cond_var)

            a_t = self.process.alpha(t)
            a_s = self.process.alpha(s)
            decay = math.exp(- self.process.mean_rev * dt)
            E_rt = rates[i-1] * decay + a_t - a_s * decay
            r_t = E_rt + std_dev*W[:,i]
            rates.append(r_t)
            zb_curve_i = tf.Variable([self.process.zero_bond(t, t + pillar, r_t) for pillar in pillars_dfs])
            zb_tensor.append(zb_curve_i)
        zb_tensor = tf.Variable(zb_tensor)
        rates = tf.Variable(rates)

        return rates, zb_tensor, time_grid



class DiscountCurveSimple:
    def __init__(self, pillars: list[float], rates: list[float]):
        self.pillars = pillars 
        self.discount_factors = tf.Variable(rates, dtype=dtypes.float64)     # tensor list

    def discount(self, term: float):
        if term <= self.pillars[0]:
            first_df = self.discount_factors[0]
            return first_df
        if term >= self.pillars[-1]:
            last_df =  self.discount_factors[-1]
            return last_df
        for i in range(0, len(self.pillars) - 1):
            if term < self.pillars[i + 1]:
                dtr = 1 / (self.pillars[i + 1] - self.pillars[i])
                w1 = (self.pillars[i + 1] - term) * dtr
                w2 = (term - self.pillars[i]) * dtr
                r1 = w1 * self.discount_factors[i]
                r2 = w2 * self.discount_factors[i + 1]
                rm = r1 + r2
                return rm
            
    def zero_rate(self, term: float):
        compound = 1.0 / self.discount(term)
        return InterestRate.implied_rate(compound, CompoundingType.Simple, Frequency.Annual, term).rate
    
    def inst_fwd(self, t: float):
        # time-step needed for differentiation
        dt = 0.01    
        expr = - (tf.math.log(self.discount(t+dt))- tf.math.log(self.discount(t-dt)))/(2*dt)
        return expr
    
    def forward_rate(self,
                    d1,
                    d2,
                    daycounter,
                    evaluation_date):
        ''' 
        Calcola il tasso forward.
        '''
        tau = daycounter.year_fraction(d1, d2)
        df1 = self.discount(daycounter.year_fraction(evaluation_date, d1))  
        df2 = self.discount(daycounter.year_fraction(evaluation_date, d2)) 
        return (df1 / df2 -1) / tau








