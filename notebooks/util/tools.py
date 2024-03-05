import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql
import datetime
from collections import namedtuple
import xlwings as xw


def makeSwap(start, maturity, nominal, fixedRate, index, typ=ql.VanillaSwap.Payer):
    end = ql.TARGET().advance(start, maturity)
    fixedLegTenor = ql.Period("1y")
    fixedLegBDC = ql.ModifiedFollowing
    fixedLegDC = ql.Thirty360(ql.Thirty360.BondBasis)
    spread = 0.0
    fixedSchedule = ql.Schedule(start,
                                end, 
                                fixedLegTenor, 
                                index.fixingCalendar(), 
                                fixedLegBDC,
                                fixedLegBDC, 
                                ql.DateGeneration.Backward,
                                False)
    floatSchedule = ql.Schedule(start,
                                end,
                                index.tenor(),
                                index.fixingCalendar(),
                                index.businessDayConvention(),
                                index.businessDayConvention(),
                                ql.DateGeneration.Backward,
                                False)
    swap = ql.VanillaSwap(typ, 
                          nominal,
                          fixedSchedule,
                          fixedRate,
                          fixedLegDC,
                          floatSchedule,
                          index,
                          spread,
                          index.dayCounter())
    return swap, [index.fixingDate(x) for x in floatSchedule][:-1]

def makeCap(start, maturity, nominal, strike, index):
    end = ql.TARGET().advance(start, maturity)
    capfloorSchedule = ql.Schedule(start,end,
                                index.tenor(),
                                index.fixingCalendar(),
                                index.businessDayConvention(),
                                index.businessDayConvention(),
                                ql.DateGeneration.Backward,
                                False)
    ibor_leg = ql.IborLeg([nominal], capfloorSchedule, index)
    cap = ql.Cap(ibor_leg, [strike])
    return cap, [index.fixingDate(x) for x in capfloorSchedule][:-1]

def short_rate_path(time_grid):
    m = len(time_grid)
    r_t = np.zeros(m)
    numeraire = np.zeros(m)
    for i in range(1, m):
        numeraire[0] = model.numeraire(0,0)
        t0 = time_grid[i-1]
        t1 = time_grid[i]
        e = process.expectation(t0, r_t[i-1], dt[i-1])
        std = process.stdDeviation(t0, r_t[i-1], dt[i-1])
        r_t[i] = np.random.normal(loc= e, scale= std)
        numeraire[i] = model.numeraire(t1, r_t[i])
    return r_t, numeraire
def ratesSimulation(nPath, time_grid):
    m = len(time_grid)
    r_t = np.zeros((nPath, m))
    numeraires = np.zeros((nPath, m))
    swapNPV = np.zeros((nPath, m))
    flt = np.zeros((nPath, m))
    fix = np.zeros((nPath, m))
    for i in range(nPath):
        r_t[i], numeraires[i]= short_rate_path(time_grid)
    return r_t, numeraires

# calibration functions
def create_swaption_helpers(data, index, term_structure, engine):
    swaptions = []
    fixed_leg_tenor = ql.Period(1, ql.Years)
    fixed_leg_daycounter = ql.Actual360()
    floating_leg_daycounter = ql.Actual360()
    for d in data:
        vol_handle = ql.QuoteHandle(ql.SimpleQuote(d.volatility))
        helper = ql.SwaptionHelper(d.start,
                                    d.length,
                                    vol_handle,
                                    index,
                                    fixed_leg_tenor,
                                    fixed_leg_daycounter,
                                    floating_leg_daycounter,
                                    term_structure,
                                    ql.BlackCalibrationHelper.RelativePriceError,
                                    ql.nullDouble(),
                                    1.0,
                                    ql.Normal,
                                    0.00 #shift to make rates non-negative
                                  )
                                   
        helper.setPricingEngine(engine)
        swaptions.append(helper)
    return swaptions    

def calibration_report(swaptions, data):
    print("-"*82)
    print("%15s %15s %15s %15s %15s" % \
    ("Model Price", "Market Price", "Implied Vol", "Market Vol", "Rel Error"))
    print( "-"*82)
    cum_err = 0.0
    implvol = []
    for i, s in enumerate(swaptions):
        model_price = s.modelValue()
        market_vol = data[i].volatility
        black_price = s.blackPrice(market_vol)
        rel_error = model_price/black_price - 1.0
        implied_vol = s.impliedVolatility(model_price,
                                          1e-5, 50, -0.0001, 2.0)
        rel_error2 = implied_vol/market_vol-1.0
        cum_err += rel_error2*rel_error2
        implvol.append(implied_vol)
        print( "%15.5f %15.5f %15.5f %15.5f %15.5f" % \
        (model_price, black_price, implied_vol, market_vol, rel_error))
    print( "-"*82)
    print( "Cumulative Error : %15.5f" % np.sqrt(cum_err))
    
    return implvol

def cleanData(eur6m_bpvols_data):
    eur6m_oswpVol = eur6m_bpvols_data/1e4
    # change to ql.Period object
    
    
    start = [ql.Period("1m"), ql.Period("2m"), ql.Period("3m"), 
             ql.Period("6m"), ql.Period("9m"), ql.Period("1y"), 
             ql.Period("18m"), ql.Period("2y"), ql.Period("3y"), 
             ql.Period("4y"), ql.Period("5y"), ql.Period("6y"),
             ql.Period("7y"), ql.Period("10y"), ql.Period("12y"),
             ql.Period("15y"), ql.Period("20y"), ql.Period("25y"), ql.Period("30y")
            ]
    end = [ql.Period("1y"), ql.Period("2y"), ql.Period("3y"), 
            ql.Period("4y"), ql.Period("5y"), ql.Period("6y"),
            ql.Period("7y"),  ql.Period("8y"), ql.Period("9y"),
            ql.Period("10y"), ql.Period("15y"), ql.Period("20y"), ql.Period("30y")
          ]
    eur6m_oswpVol.index = start
    eur6m_oswpVol.columns = end
    return eur6m_oswpVol

def calibrateGSR(eur6m_bpvols_data, today, t0_curve, index, calendar):
    eur6m_oswpVol = cleanData(eur6m_bpvols_data)
    # calibration dates
    CalibrationData = namedtuple("CalibrationData", 
                                 "start, length, volatility")
    startCalibration = eur6m_oswpVol.index.values

    endCalibration = [ql.Period("30y"), ql.Period("20y"), ql.Period("15y"), 
                        ql.Period("15y"), ql.Period("10y"), ql.Period("10y"), 
                        ql.Period("9y"), ql.Period("8y"), ql.Period("7y"), 
                        ql.Period("6y"), ql.Period("6y"), ql.Period("5y"),
                        ql.Period("5y"), ql.Period("4y"), ql.Period("4y"),
                        ql.Period("3y"), ql.Period("3y"), ql.Period("2y"), ql.Period("1y")
                       ]
    calibrationBasket = []
    for s,e in zip(startCalibration, endCalibration):
        calibrationBasket += [CalibrationData(s, e, eur6m_oswpVol.loc[s,e])]
    stepDate_calibration = [calendar.advance(today, d) for d in startCalibration]
    stepDate_calibration = stepDate_calibration[1:-1] # togliere la prima e ultima data
    
    sigmas = [ql.QuoteHandle(ql.SimpleQuote(0.01))
          for x in range(0, len(stepDate_calibration)+1)] # n+1 vols
    reversion = [ql.QuoteHandle(ql.SimpleQuote(0.01))]
    model = ql.Gsr(t0_curve, stepDate_calibration, sigmas, reversion,  400.)
    process = model.stateProcess()
    engine = ql.Gaussian1dSwaptionEngine(model)
    oswp_helpers = create_swaption_helpers(calibrationBasket, index, t0_curve, engine)
    optimization_method = ql.LevenbergMarquardt()
    end_criteria = ql.EndCriteria(1000, 10, 1e-8, 1e-8, 1e-8)
    model.calibrate(oswp_helpers, optimization_method, end_criteria)
    report = calibration_report(oswp_helpers, calibrationBasket)
    return model

def helpersMatr(model, eur6m_bpvols_data, t0_curve, index):
    # out of sample test calibrazione
    eur6m_oswpVol = cleanData(eur6m_bpvols_data)
    start = eur6m_oswpVol.index.values
    end = eur6m_oswpVol.columns.values
    engine = ql.Gaussian1dSwaptionEngine(model)
    helpersmatr = []
    for i,s in enumerate(start):
        helpersrow = []
        fixed_leg_tenor = ql.Period(1, ql.Years)
        fixed_leg_daycounter = ql.Actual360()
        floating_leg_daycounter = ql.Actual360()
        for j,e in enumerate(end):
            vol_handle = ql.QuoteHandle(ql.SimpleQuote(eur6m_oswpVol.loc[s,e]))
            helper = ql.SwaptionHelper(s,
                                        e,
                                        vol_handle,
                                        index,
                                        fixed_leg_tenor,
                                        fixed_leg_daycounter,
                                        floating_leg_daycounter,
                                        t0_curve,
                                        ql.BlackCalibrationHelper.RelativePriceError,
                                        ql.nullDouble(),
                                        1.0,
                                        ql.Normal,
                                        0.00 )
            helper.setPricingEngine(engine)
            helpersrow.append(helper)
        helpersmatr.append(helpersrow)
    return helpersmatr