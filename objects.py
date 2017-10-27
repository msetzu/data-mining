import pandas as pd


class HR:
	def __init__(self, data, normalised=False):
		"""
		Constructor.
		:param data:		The pandas dataframe to use to construct the object.
		:param normalised: 	If True, data.normalised contains a normalised version of the data columns.
		"""
		self.data = {}
		self.data["satisfaction_level"] = data["satisfaction_level"]
		self.data["last_evaluation"] = data["last_evaluation"]
		self.data["number_project"] = data["number_project"]
		self.data["average_montly_hours"] = data["average_montly_hours"]
		self.data["time_spend_company"] = data["time_spend_company"]
		self.data["Work_accident"] = data["Work_accident"]
		self.data["left"] = data["left"]
		self.data["promotion_last_5years"] = data["promotion_last_5years"]
		self.data["sales"] = data["sales"]
		self.data["salary"] = data["salary"]
		self.data = pd.DataFrame.from_dict(self.data)


		self.normal = {}
		self.normal["satisfaction_level"] = (data["satisfaction_level"] - data["satisfaction_level"].mean()) / data["satisfaction_level"].std()
		self.normal["last_evaluation"] = data["last_evaluation"]
		self.normal["number_project"] = (data["number_project"] - data["number_project"].mean()) / data["number_project"].std()
		self.normal["average_montly_hours"] = (data["average_montly_hours"] - data["average_montly_hours"].mean()) / data["average_montly_hours"].std()
		self.normal["time_spend_company"] = (data["time_spend_company"] - data["time_spend_company"].mean()) / data["time_spend_company"].std()
		self.normal["Work_accident"] = data["Work_accident"]
		self.normal["left"] = data["left"]
		self.normal["promotion_last_5years"] = data["promotion_last_5years"]
		self.normal["sales"] = data["sales"]
		self.normal["salary_int"] = data.assign(salary_int=self.salaries(data))["salary_int"]
		self.normal["salary_int"] = (self.normal["salary_int"] - self.normal["salary_int"].mean()) / self.normal["salary_int"].std()
		self.normal = pd.DataFrame.from_dict(self.normal)

		self.discrete = {}
		self.discrete["satisfaction_level"] = data.assign(satisfaction_level=round(data["satisfaction_level"] * 10).astype(int))["satisfaction_level"]
		self.discrete["last_evaluation"] = data.assign(last_evaluation=round(data["last_evaluation"] * data["time_spend_company"]).astype(int))["last_evaluation"]
		self.discrete["number_project"] = self.data["number_project"]
		self.discrete["average_montly_hours"] = self.data["average_montly_hours"].astype(int)
		self.discrete["time_spend_company"] = self.data["time_spend_company"]
		self.discrete["Work_accident"] = self.data["Work_accident"]
		self.discrete["left"] = self.data["left"]
		self.discrete["promotion_last_5years"] = self.data["promotion_last_5years"]
		self.discrete["sales"] = self.data["sales"]
		self.discrete["salary"] = self.data["salary"]
		self.discrete["salary_int"] = data.assign(salary_int = self.salaries(data))["salary_int"]
		self.discrete = pd.DataFrame.from_dict(self.discrete)

		self.std = {}
		self.std["satisfaction_level"] = data["satisfaction_level"].std().astype(list)
		self.std["last_evaluation"] = data["last_evaluation"].std().astype(list)
		self.std["number_project"] = data["number_project"].std().astype(list)
		self.std["average_montly_hours"] = data["average_montly_hours"].std().astype(list)
		self.std["time_spend_company"] = data["time_spend_company"].std().astype(list)
		self.std["Work_accident"] = data["Work_accident"].std().astype(list)
		self.std["left"] = data["left"].std().astype(list)
		self.std["promotion_last_5years"] = data["promotion_last_5years"].std().astype(list)

		self.mean = {}
		self.mean["satisfaction_level"] = data["satisfaction_level"].mean().astype(list)
		self.mean["last_evaluation"] = data["last_evaluation"].mean().astype(list)
		self.mean["number_project"] = data["number_project"].mean().astype(list)
		self.mean["average_montly_hours"] = data["average_montly_hours"].mean().astype(list)
		self.mean["time_spend_company"] = data["time_spend_company"].mean().astype(list)
		self.mean["Work_accident"] = data["Work_accident"].mean().astype(list)
		self.mean["left"] = data["left"].mean().astype(list)
		self.mean["promotion_last_5years"] = data["promotion_last_5years"].mean().astype(list)

	def salaries(self, data):
		salaries = list(data["salary"])
		salaries_discrete = {"low": 10000, "medium": 25000, "high": 50000}
		return list(map(lambda x: salaries_discrete[x], salaries))

