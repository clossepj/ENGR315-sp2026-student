import pandas as pd
import numpy as np
from scipy.stats import norm, chisquare, ttest_ind, ttest_1samp
import matplotlib.pyplot as plt

"""
Preamble: Load data from source CSV file
"""
path_to_datafile = "../../data/drop-jump/all_participant_data_rsi.csv"
df = pd.read_csv(path_to_datafile)
print(df.head())
print(df.columns)

accel = df["accelerometer_rsi"].dropna()
fp = df["force_plate_rsi"].dropna()


"""
Question 1: Load the force plate and acceleration based RSI data for all participants. Map each data set (accel and FP)
to a normal distribution. Clearly report the distribution parameters (mu and std) and generate a graph two each curve's 
probability distribution function. Include appropriate labels, titles, and legends.
"""
print('-----Question 1-----')

### YOUR CODE HERE
mu_accel = np.mean(accel)
std_accel = np.std(accel)

mu_fp = np.mean(fp)
std_fp = np.std(fp)
##Calculations for mean and standard deviation, print functions below. 

print("mu accelerometer | mu force plate | std accelerometer | std force plate")
print(mu_accel, "|", mu_fp, "|", std_accel, "|", std_fp)


#Plotting Probability Distribution Function for Accelerometer and Force Plate
x_accel = np.linspace(min(accel), max(accel), 1000) #RSI Values
y_accel = norm.pdf(x_accel, mu_accel, std_accel) #Probability Density
x_fp = np.linspace(min(fp), max(fp), 1000)
y_fp = norm.pdf(x_fp, mu_fp, std_fp)
plt.plot(x_accel, y_accel, label='Accelerometer PDF')
plt.plot(x_fp, y_fp, label='Force Place PDF')
plt.title("RSI Normal Distribution")
plt.xlabel("RSI")
plt.ylabel("Probability Density")
plt.legend()
plt.show()

"""
Question 2: Conduct a Chi2 Goodness of Fit Test for each dataset to test whether the data is a good fit
for the derived normal distribution. Clearly print out the p-value, chi2 stat, and an indication of whether it is 
a fit or not. Do this for both acceleration and force plate distributions. It is suggested to generate 9 bins between 
[0,2), add append -inf and +inf to both ends of the bins. An alpha=0.05 is suitable for these tests.
"""
print('\n\n-----Question 2-----')

"""
Acceleration
"""
### YOUR CODE HERE
# making the bins
bins = list(np.linspace(0, 2, 10)) #9 main bins
bins.append(np.inf) #1 additional infinity bin

#expected counts variables
observed, _ = np.histogram(accel, bins=bins)
cdf_values = norm.cdf(bins, mu_accel, std_accel)
probabilities = np.diff(cdf_values)
expected = probabilities * len(accel)
expected = expected * (sum(observed) / sum(expected)) #normalizes expected counts to prevent chi square errors 
chi2_stat, p_value = chisquare(observed, expected) # actual chi square calculation

#print script of data in terminal, takes alpha into account for good/not good fit
print("Acceleration Data:")
print("Chi2 Stat:", chi2_stat)
print("p_value:", p_value)
if p_value > 0.05:
    print("Good Fit\n")
else:
    print("Not a Good Fit\n")

"""
Force Plate
"""
### YOUR CODE HERE
#expected counts variables
observed_fp, _ = np.histogram(fp, bins=bins)
cdf_values_fp = norm.cdf(bins, mu_fp, std_fp)
probabilities = np.diff(cdf_values_fp)
expected_fp = probabilities * len(accel)
expected_fp = expected_fp * (sum(observed_fp) / sum(expected_fp)) #normalizes expected counts to prevent chi square errors 
chi2_stat_fp, p_value_fp = chisquare(observed_fp, expected_fp) # actual chi square calculation

#print script of data in terminal, takes alpha into account for good/not good fit
print("Force Plate Data:")
print("Chi2 Stat:", chi2_stat_fp)
print("p_value:", p_value_fp)
if p_value_fp > 0.05:
    print("Good Fit\n")
else:
    print("Not a Good Fit\n")


"""
Question 3: Perform a t-test to determine whether the RSI means for the acceleration and force plate data are equivalent 
or not. Clearly report the p-value for the t-test and make a clear determination as to whether they are equal or not.
An alpha=0.05 is suitable for these tests.
"""
print('\n\n-----Question 3-----')

### YOUR CODE HERE
t_stat, p_value = ttest_ind(accel, fp) #this is simply the 2 sample t-test
#this simply prints the response from the t-test, and compares to alpha
if p_value > 0.05:
    print("Means are Statistically Equal")
else:
    print("Means are Statistically Different")
print("-" * 45)


"""
Question 4: Calculate the RSI Error for the dataset where error is expressed as the difference between the 
Force Plate RSI measurement and the Accelerometer RSI measurement. Fit this error distribution to a normal curve and 
plot a histogram of the data on the same plot showing the fitted normal curve. Include appropriate labels, titles, and 
legends. The default binning approach from matplot lib with 16 bins is sufficient.
"""

### YOUR CODE HERE
error = fp - accel #error calculation
print("Error=", error)

#below procedure is essentialy the same as Q1, just this time utilizing the error values 
mu_error = np.mean(error)
std_error = np.std(error)

#histogram plotting script
plt.hist(error, bins=16, density=True, alpha = 0.6)

x = np.linspace(min(error), max(error), 1000)
y = norm.pdf(x, mu_error, std_error)

plt.plot(x, y, label="Normal Fitted Curve")
plt.title("RSI Error")
plt.xlabel("Error [fp - accel]")
plt.ylabel("Density")
plt.legend()
plt.show()