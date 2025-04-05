

downloadFiles:
	pip install med-minds
	python3 main.py

initServer:
	sudo docker run -d --name minds -e POSTGRES_PASSWORD=minds-secure-password -e POSTGRES_DB=minds -p 5433:5432 postgres
