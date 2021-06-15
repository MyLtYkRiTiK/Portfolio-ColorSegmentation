#!/bin/bash
docker build -t colorsegmentationflaskimg -f Dockerfile .
docker-compose up -d