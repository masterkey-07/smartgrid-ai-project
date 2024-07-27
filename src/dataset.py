from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
electrical_grid_stability_simulated_data = fetch_ucirepo(id=471) 
  
# data (as pandas dataframes) 
X = electrical_grid_stability_simulated_data.data.features 
y = electrical_grid_stability_simulated_data.data.targets 
  
# metadata 
print(electrical_grid_stability_simulated_data.metadata) 
  
# variable information 
print(electrical_grid_stability_simulated_data.variables)