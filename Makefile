.PHONY: pull build pull_model up down restart logs clean ps

pull:
	docker compose pull

build:
	docker compose build

pull_model:
    @if [ -z "$(model)" ]; then echo "Please specify a model, e.g., make pull_model model=llama3.2"; exit 1; fi
	docker compose exec -it ollama ollama pull $(model)
	
up:
	docker compose up -d

down:
	docker compose down

restart: down up

logs:
	docker compose logs -f

clean:
	docker compose down -v --rmi all

ps:
	docker compose ps

clean-volumes:
	docker compose down -v

restart-log-analyzer:
	docker compose restart log-analyzer

restart-ollama:
	docker compose restart ollama

restart-chromadb:
	docker compose restart chromadb