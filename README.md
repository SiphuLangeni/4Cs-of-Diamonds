# 4Cs of Diamonds

The goal of this project is to build a regression model that can predict the price of round cut diamonds. The model will be deployed using FastAPI and heroku.

## Creating the Dataset

Diamond data was obtained from brilliantearth.com using the requests library. A diamond scraper class was created to parse the json for relevant features, enforce dtypes and save to a csv.

## About the dataset

47,210 record with 15 columns

**upc:** Unique identifier  
**cut:** Quality of the cut (Fair, Good, Very Good, Ideal, Super Ideal)  
**colour:** Diamond color (J, I, H, G, F, E, D)  
**clarity:** Amount of inclusions (SI2, SI1, VS2, VS1, VVS2, VVS1, FL, IF)  
**carat:** Weight of the diamond (0.25 - 1.0)  
**x:** Length in mm (3.97 - 6.65)  
**y:** Width in mm (3.93 - 6.62)  
**z:** Depth in mm (2.41 - 4.21)  
**lw_ratio:** Ratio x:y (1.00 - 1.03)  
**depth:** Depth as a percentage (56.1 - 71.2)  
**table:** Width of the diamond relative to the widest point (41.4 - 71.0)    
**fluorescence:** Glow when subject to UV light (Very Strong, Strong, Slight, Medium, Very Slight, Faint, None)   
**report:** Lab of certification (GIA, IGI, HRD)  
**origin:** Place of origin of the diamond (Botswana Sort, Russia, Canada, Recycled)  
**price:** Price in CAD ($540 - $23,300)

## Deploying the model
Model endpoint was created using FastAPI and deployed using Heroku.

## Streamlit App
A simple Streamlit app was built to show the model in action!



