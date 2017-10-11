class HR:
	def __init__(self, data):
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

		self.std = {}
		self.std["satisfaction_level"] = self.data["satisfaction_level"].std()
		self.std["last_evaluation"] = self.data["last_evaluation"].std()
		self.std["number_project"] = self.data["number_project"].std()
		self.std["average_montly_hours"] = self.data["average_montly_hours"].std()
		self.std["time_spend_company"] = self.data["time_spend_company"].std()
		self.std["Work_accident"] = self.data["Work_accident"].std()
		self.std["left"] = self.data["left"].std()
		self.std["promotion_last_5years"] = self.data["promotion_last_5years"].std()

		self.mean = {}
		self.mean["satisfaction_level"] = self.data["satisfaction_level"].mean()
		self.mean["last_evaluation"] = self.data["last_evaluation"].mean()
		self.mean["number_project"] = self.data["number_project"].mean()
		self.mean["average_montly_hours"] = self.data["average_montly_hours"].mean()
		self.mean["time_spend_company"] = self.data["time_spend_company"].mean()
		self.mean["Work_accident"] = self.data["Work_accident"].mean()
		self.mean["left"] = self.data["left"].mean()
		self.mean["promotion_last_5years"] = self.data["promotion_last_5years"].mean()