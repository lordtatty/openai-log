package gptlog

import (
	"context"
	"fmt"
	"log"

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
	Client       *openai.Client
	Usage        Usage
	DefaultModel string
}

func (a *AI) ChatCompletion(r openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
	ctx := context.Background()
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
	return &resp, nil
}

func (a *AI) Embeddings(r openai.EmbeddingRequest) (*openai.EmbeddingResponse, error) {
	ctx := context.Background()
	resp, err := a.Client.CreateEmbeddings(ctx, r)
	if err != nil {
		return nil, fmt.Errorf("Embeddings error: %v", err)
	}
	a.Usage.Add(string(r.Model), resp.Usage)
	return &resp, nil
}