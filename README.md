# Automatic Vehicle Detection and Tracking in Aerial Video #

## Abstract ##

In	 this	 report,	we	present	an	approach	 for	 the	automatic	detection	and	 tracking	of	
vehicles	in	aerial	video.	The	proposed	system	is	comprised	of	 three	stages	namely;	
Region	 of	Interest (ROI) Selection,	 Classification	and	Tracking.	In	 the	 first	 stage	we	
expedite	 detection	 by	 identifying	 image	 regions	 that	 are	 most	 likely	 to	 contain	
vehicles.	 To	 achieve	 this	 we	 use	 a	 fast	 corner	 detector	 combined	 with	 an efficient	
feature	 density	 estimation	 technique.	 In	 the	 classification	 stage	 the	 system	 uses	
shape	 encoding	 local	 features	 and	 robust	machine	 learning	 techniques	 to	 perform	
efficient	vehicle	detection.	In	the	final	stage we	propose	a	tracking	algorithm,	which
employs	particle	 filtering	 to	exploit	vehicle	dynamics,	and	improve	tracker-detector	
associations.	 We	 evaluate	 the	 proposed	 system	 by	 performing	 a	 number	 of	
experiments	and	use	 the	 results	 to	show	 that	 the	system	is	capable	of	successfully	
detecting	 and	 tracking	 a variety	 of differing	 vehicle	 types	 under	 varying	 rotation,	
sheering and	blurring	conditions.
