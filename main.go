package main

import (
	"net/http"
	"time"

	"strings"

	"context"
	"fmt"
	"log"
	"os"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"

	chroma_go "github.com/amikos-tech/chroma-go/types"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/chroma"

	"github.com/tmc/langchaingo/llms/openai"
)

// Define the payload structure
type CoverLetterRequest struct {
	Template       string   `json:"template"`
	BodyParagraphs []string `json:"body_paragraphs"`
	JobDescription string   `json:"job_description"`
}

type EmbeddingResponse struct {
	Embeddings []float64 `json:"embeddings"`
}

// This receives a CoverLetterRequest and Generates a Cover Letter.
func handleGenerateCoverLetter(c *gin.Context) {
	var req CoverLetterRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request payload"})
		return
	}

	// Validate that none of the fields are empty
	if req.Template == "" || req.BodyParagraphs == nil || req.JobDescription == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request payload. All fields must be provided"})
		return
	}

	// Validate that there are at least 3 body paragraphs
	if len(req.BodyParagraphs) < 3 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request payload. There must be at least 3 body paragraphs"})
		return
	}

	// Validate that the template contains the {experiences} placeholder
	if !strings.Contains(req.Template, "{experiences}") {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request payload. template should contain '{experiences}'"})
		return
	}

	// pick 3 most relevant paragraphs
	pars, err := choose_3_pars_w_vectorstore(req.BodyParagraphs, req.JobDescription)
	// pars, err := choose_3_pars(req.BodyParagraphs, req.JobDescription)
	if err != nil {
		log.Println(err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failure choosing 3 body paragraphs"})
	}
	// fmt.Println("3 most relevant paragraphs:", pars)

	// fill out template
	coverLetterPartial, err := fill_template(req.Template, req.JobDescription)
	if err != nil {
		log.Println(err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failure filling out template"})
	}
	// fmt.Println("coverLetterPartial:\n", coverLetterPartial)

	// Replace placeholder with the actual paragraphs
	coverLetter := strings.Replace(coverLetterPartial, "{experiences}", pars, 1)

	response := gin.H{
		"cover_letter": coverLetter,
	}
	// fmt.Println("response:", response)
	c.JSON(http.StatusOK, response)
}

func getFormattedDate() string {
	now := time.Now()
	format := "January 2, 2006"
	formattedDate := now.Format(format)
	return formattedDate
}

// fill_template generates a cover letter by filling out missing fields in the provided template
// using the job description. It sends the combined prompt to an LLM model and returns the
// generated cover letter or an error if the operation fails.
//
// Parameters:
//   - template: A string containing the cover letter template with placeholders for missing fields.
//   - jobDescription: A string containing the job description to be used for filling in the template.
//
// Returns:
//   - A string containing the generated cover letter with fields filled in based on the job description.
//   - An error if there was an issue loading the model or generating the completion.
func fill_template(template string, jobDescription string) (string, error) {
	modelName := "gpt-4o-mini"
	llm, err := openai.New(openai.WithModel(modelName))
	if err != nil {
		return "", fmt.Errorf("failed to load model: %w", err)
	}
	ctx := context.Background()

	const systemPrompt = "Use the job description to fill out all missing fields like these: <MISSING_FIELD>. If there is a field with curly brackets like this {example}, leave it unmodified in your output. Your output should not include the job description."
	date := getFormattedDate()
	combinedPrompt := fmt.Sprintf("%s\n\nToday's date:%s\n\nJob Description:\n%s\n\nTemplate:\n%s", systemPrompt, date, jobDescription, template)
	// fmt.Println("combinedPrompt:\n", combinedPrompt)
	completion, err := llm.Call(ctx, combinedPrompt,
		llms.WithTemperature(0.8),
	)
	if err != nil {
		return "", fmt.Errorf("failed to generate completion: %w", err)
	}
	return completion, nil
}

// choose_3_pars chooses 3 most relevant body paragraphs for the job description from
// a list of body paragraphs. It uses an LLM prompt with context.
//
// Parameters:
//   - bodyPars: A lsit of strings containing the body paragraphs to choose from.
//   - jobDescription: A string containing the job description to be used for filling in the template.
//
// Returns:
//   - A list of 3 strings containing the body pars.
//   - An error, if any errors occur.
func choose_3_pars(bodyPars []string, jobDescription string) (string, error) {
	modelName := "gpt-4o-mini"
	llm, err := openai.New(openai.WithModel(modelName))
	if err != nil {
		return "", fmt.Errorf("failed to load model: %w", err)
	}
	ctx := context.Background()

	// Format the body paragraphs into a single string for the prompt
	paragraphs := ""
	for i, paragraph := range bodyPars {
		paragraphs += fmt.Sprintf("%d. %s\n", i+1, paragraph)
	}

	const systemPrompt = `
You are given a job description and a list of body paragraphs. Your task is to select the three most relevant body paragraphs for the job description. Relevance should be based on how well the paragraph matches the requirements and details mentioned in the job description.

Job Description:
%s

Body Paragraphs:
%s

Please return a list of the three most relevant body paragraphs in the following format:

1. First most relevant paragraph.
2. Second most relevant paragraph.
3. Third most relevant paragraph.
`
	combinedPrompt := fmt.Sprintf(systemPrompt, jobDescription, paragraphs)
	// fmt.Println("combinedPrompt:\n", combinedPrompt)
	completion, err := llm.Call(ctx, combinedPrompt,
		llms.WithTemperature(0.8),
	)
	if err != nil {
		return "", fmt.Errorf("failed to generate completion: %w", err)
	}

	return completion, nil
}

// // choose_3_pars_with_vectorstore chooses 3 most relevant body paragraphs for the job description from
// // a list of body paragraphs. It uses a vectorstore and a query to select the 3 most relevant body paragraphs.
// //
// // Parameters:
// //   - bodyPars: A lsit of strings containing the body paragraphs to choose from.
// //   - jobDescription: A string containing the job description to be used for filling in the template.
// //
// // Returns:
// //   - The 3 body paragraphs, as a single string, separated by newlines.
// //   - An error, if any errors occur.
// func choose_3_pars_with_vectorstore(bodyPars []string, jobDescription string, caseToRun int) (string, error) {
// 	store := createVectorStore(bodyPars)
// 	ctx := context.TODO()

// 	type exampleCase struct {
// 		name         string
// 		query        string
// 		numDocuments int
// 		options      []vectorstores.Option
// 	}

// 	// type filter = map[string]any
// 	jobDescriptionSummarized := jobDescription

// 	cases := []exampleCase{
// 		{
// 			name:         "Full Job Description",
// 			query:        fmt.Sprintf("Which of these paragraphs is most related to the job description?\n\nJob Description:{%s}", jobDescription),
// 			numDocuments: 3,
// 			options: []vectorstores.Option{
// 				vectorstores.WithScoreThreshold(0.8),
// 			},
// 		},
// 		{
// 			name:         "Summarized Job Description",
// 			query:        fmt.Sprintf("Which of these paragraphs is most related to the job description?\n\nJob Description:{%s}", jobDescriptionSummarized),
// 			numDocuments: 3,
// 			options: []vectorstores.Option{
// 				vectorstores.WithScoreThreshold(0.8),
// 			},
// 		},
// 	}

// 	ec := cases[caseToRun]
// 	docs, err := store.SimilaritySearch(ctx, ec.query, ec.numDocuments, ec.options...)
// 	if err != nil {
// 		log.Fatalf("query: %v\n", err)
// 	}
// 	fmt.Println("docs:", docs)
// 	texts := make([]string, len(docs))
// 	for docI, doc := range docs {
// 		texts[docI] = doc.PageContent
// 	}

// 	fmt.Printf("%d. case: %s\n", caseToRun, ec.name)
// 	fmt.Printf("    result: %s\n", strings.Join(texts, "\n"))
// 	return strings.Join(texts, "\n"), nil

// 	// // print out the results of the run
// 	// fmt.Printf("Results:\n")
// 	// for ecI, ec := range cases {
// 	// 	texts := make([]string, len(results[ecI]))
// 	// 	for docI, doc := range results[ecI] {
// 	// 		texts[docI] = doc.PageContent
// 	// 	}
// 	// 	fmt.Printf("%d. case: %s\n", ecI+1, ec.name)
// 	// 	fmt.Printf("    result: %s\n", strings.Join(texts, ", "))
// 	// }

// }

func choose_3_pars_w_vectorstore(bodyPars []string, jobDescription string) (string, error) {
	store, errNs := chroma.New(
		chroma.WithChromaURL(os.Getenv("CHROMA_URL")),
		chroma.WithOpenAIAPIKey(os.Getenv("OPENAI_API_KEY")),
		chroma.WithDistanceFunction(chroma_go.COSINE),
		chroma.WithNameSpace(uuid.New().String()),
	)
	if errNs != nil {
		log.Fatalf("new: %v\n", errNs)
	}

	type meta = map[string]any

	// Create an array of schema.Document
	var documents []schema.Document

	// Iterate over bodyPars and add them as documents
	for _, bodyPar := range bodyPars {
		doc := schema.Document{
			PageContent: bodyPar,
			Metadata:    meta{"tmpkey": "tmpval"}, // Example of adding metadata
		}
		documents = append(documents, doc)
	}

	// Add documents to the vector store
	_, err := store.AddDocuments(context.Background(), documents)
	if err != nil {
		log.Fatal("Error adding documents to the store: ", err)
	}

	log.Println("Documents added successfully.")

	ctx := context.TODO()

	type exampleCase struct {
		name         string
		query        string
		numDocuments int
		options      []vectorstores.Option
	}

	type filter = map[string]any

	exampleCases := []exampleCase{
		{
			name:         "Software Dev",
			query:        fmt.Sprintf("Which of these paragraphs is most related to the job description?\n\nJob Description: %s", jobDescription),
			numDocuments: 3,
			options: []vectorstores.Option{
				vectorstores.WithScoreThreshold(0.8),
			},
		},
		{
			name:         "Data Engineering",
			query:        "Which of these paragraphs is most related to Data Engineering?",
			numDocuments: 3,
			options: []vectorstores.Option{
				vectorstores.WithScoreThreshold(0.8),
			},
		},
		{
			name:         "NLP",
			query:        "Which of these paragraphs is most related to NLP (Natual Language Processing)?",
			numDocuments: 3,
			options: []vectorstores.Option{
				vectorstores.WithFilters(filter{
					"$and": []filter{
						{"area": filter{"$gte": 1000}},
						{"population": filter{"$gte": 13}},
					},
				}),
			},
		},
	}

	const caseToRun = 1
	ec := exampleCases[caseToRun]
	docs, err := store.SimilaritySearch(ctx, ec.query, ec.numDocuments, ec.options...)
	if err != nil {
		log.Fatalf("query: %v\n", err)
	}
	texts := make([]string, 3)
	for docI, doc := range docs {
		texts[docI] = doc.PageContent
	}
	fmt.Printf("    result: %s\n", strings.Join(texts, "\n\n"))
	result := strings.Join(texts, "\n\n")
	return result, nil
}

func main() {

	// Create a new Gin router
	router := gin.Default()

	// Define the /cover-letter route and associate it with the handler
	router.POST("/cover-letter", handleGenerateCoverLetter)

	// Start the server on port 8080
	port := ":8080"
	router.Run(port)
}
