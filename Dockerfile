FROM node:12

ARG NODE_ENV=production
ENV NODE_ENV=${NODE_ENV}

WORKDIR /usr/src/app

ENV PORT 8080
ENV HOST 0.0.0.0

COPY package*.json ./

RUN npm install

# Copy the local code to the container
COPY . .


# Start the service
CMD ["npm","run","start:prod"]