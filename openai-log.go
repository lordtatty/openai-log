package gptlog

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"time"

	"github.com/sashabaranov/go-openai"
)

type ModelUsage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

type Usage struct {
	u map[string]*ModelUsage
}

type UsageCost struct {
	Model                    string
	PromptCostPerMillion     float64
	CompletionCostPerMillion float64
}

var usageCosts = map[string]UsageCost{
	openai.GPT4oMini: {
		PromptCostPerMillion:     0.150,
		CompletionCostPerMillion: 0.60,
	},
	string(openai.SmallEmbedding3): {
		PromptCostPerMillion: 0.02,
	},
}

func (u *Usage) Add(model string, r openai.Usage) error {
	if u.u == nil {
		u.u = make(map[string]*ModelUsage)
	}
	if _, ok := u.u[model]; !ok {
		u.u[model] = &ModelUsage{}
	}
	u.u[model].PromptTokens += r.PromptTokens
	u.u[model].CompletionTokens += r.CompletionTokens
	u.u[model].TotalTokens += r.TotalTokens
	return nil
}

func (u *Usage) PrintUsage() {
	totalPromptTokens := 0
	totalCompletionTokens := 0
	totalTotalTokens := 0
	totalCost := 0.0
	for k, v := range u.u {
		cost := u.cost(k)
		totalCost += cost
		fmt.Println("Model: ", k)
		fmt.Println("Prompt Tokens: ", v.PromptTokens)
		fmt.Println("Completion Tokens: ", v.CompletionTokens)
		fmt.Println("Total Tokens: ", v.TotalTokens)
		fmt.Printf("Cost: $%.9f\n", cost)
		fmt.Println("---")
		totalPromptTokens += v.PromptTokens
		totalCompletionTokens += v.CompletionTokens
		totalTotalTokens += v.TotalTokens
	}
	fmt.Println("Total Prompt Tokens: ", totalPromptTokens)
	fmt.Println("Total Completion Tokens: ", totalCompletionTokens)
	fmt.Println("Total Tokens: ", totalTotalTokens)
	fmt.Printf("Total Cost: $%.9f\n", totalCost)
}

func (u *Usage) cost(model string) float64 {
	if _, ok := u.u[model]; !ok {
		return 0
	}
	cost := float64(u.u[model].PromptTokens) * usageCosts[model].PromptCostPerMillion / 1000000
	cost += float64(u.u[model].CompletionTokens) * usageCosts[model].CompletionCostPerMillion / 1000000
	return cost
}

type AI struct {
	Client        *openai.Client
	Usage         Usage
	DefaultModel  string
	EnableLogging bool
	logFile       *os.File
}

func (a *AI) initLogFile() error {
	if a.logFile != nil {
		return nil
	}
	timestamp := time.Now().Format("20060102-150405")
	filename := fmt.Sprintf("ai-log-%s.log", timestamp)
	file, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return err
	}
	a.logFile = file
	return nil
}

func (a *AI) logJSON(data interface{}) error {
	// Convert to JSON
	entryJSON, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("error marshaling log entry to JSON: %v", err)
	}

	// Write to log file
	if a.logFile != nil {
		_, err = a.logFile.WriteString("==============\n")
		if err != nil {
			return err
		}
		_, err = a.logFile.WriteString("==============\n")
		if err != nil {
			return err
		}
		_, err = a.logFile.Write(entryJSON)
		if err != nil {
			return err
		}
		_, err = a.logFile.WriteString("\n")
		if err != nil {
			return err
		}
	}
	return nil
}

func (a *AI) logData(m []openai.ChatCompletionMessage, c []openai.ChatCompletionChoice) error {
	if !a.EnableLogging {
		return nil
	}
	if err := a.initLogFile(); err != nil {
		log.Println("Error initializing log file:", err)
		return err
	}
	return a.logJSON([]interface{}{m, c})
}

func (a *AI) CreateChatCompletion(ctx context.Context, r openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
	if r.Model == "" {
		r.Model = a.DefaultModel
	} else {
		log.Println(">>>>>>> CUSTOM MODEL USED: ", r.Model)
	}
	log.Println("Model: ", r.Model)
	resp, err := a.Client.CreateChatCompletion(ctx, r)
	if err != nil {
		return nil, fmt.Errorf("ChatCompletion error: %v", err)
	}
	a.Usage.Add(r.Model, resp.Usage)

	// Log
	if err := a.logData(r.Messages, resp.Choices); err != nil {
		log.Println("Error logging Choices:", err)
	}

	return &resp, nil
}
func (a *AI) ChatCompletionStream(ctx context.Context, r openai.ChatCompletionRequest, s chan openai.ChatCompletionStreamResponse) error {
	if r.Model == "" {
		r.Model = a.DefaultModel
	} else {
		log.Println(">>>>>>> CUSTOM MODEL USED: ", r.Model)
	}
	log.Println("Model: ", r.Model)
	if r.StreamOptions == nil {
		r.StreamOptions = &openai.StreamOptions{}
	}
	r.StreamOptions.IncludeUsage = true
	stream, err := a.Client.CreateChatCompletionStream(ctx, r)
	if err != nil {
		return fmt.Errorf("ChatCompletionStream error: %v", err)
	}
	defer stream.Close()
	choices := make([]openai.ChatCompletionChoice, 0)
	for {
		stream.Header()
		resp, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return fmt.Errorf("ChatCompletionStream error: %v", err)
		}
		s <- resp
		for i, c := range resp.Choices {
			if i > len(choices) {
				choices[i] = openai.ChatCompletionChoice{
					Index:   c.Index,
					Message: openai.ChatCompletionMessage{},
					// LogProbs: openai.ChatCompletionLogProbs{},
					// ContentFilterResults: ,
				}
				choices[i].Message.Content += c.Delta.Content
				choices[i].Message.Role += c.Delta.Role
			}
		}
		if resp.Usage != nil {
			// only the last message has the usage
			a.Usage.Add(r.Model, *resp.Usage)
		}
	}

	// Log
	if err := a.logData(r.Messages, choices); err != nil {
		log.Println("Error logging Choices:", err)
	}

	return nil
}

func (a *AI) CreateEmbeddings(ctx context.Context, r openai.EmbeddingRequestConverter) (*openai.EmbeddingResponse, error) {
	resp, err := a.Client.CreateEmbeddings(ctx, r)
	if err != nil {
		return nil, fmt.Errorf("embeddings error: %v", err)
	}
	// if r is of type openai.EmbeddingRequest then store the usage, othjerwise alert
	if cvr, ok := r.(openai.EmbeddingRequest); ok {
		a.Usage.Add(string(cvr.Model), resp.Usage)
	} else {
		log.Println("Embeddings request is not of type openai.EmbeddingRequest, so we can't store the usage")
	}
	return &resp, nil
}
