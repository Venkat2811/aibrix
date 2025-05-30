/*
Copyright 2024 The Aibrix Team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package vtc

import (
	"context"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestSlidingWindowTokenTracker_GetTokenCount(t *testing.T) {
	config := DefaultVTCConfig()
	tracker := NewInMemorySlidingWindowTokenTracker(&config, WithWindowSize(100), WithTimeUnit(Milliseconds)) // 100ms window
	ctx := context.Background()

	tokens, err := tracker.GetTokenCount(ctx, "user1")
	assert.NoError(t, err)
	assert.Equal(t, float64(0), tokens, "Initial token count should be 0")

	err = tracker.UpdateTokenCount(ctx, "user1", 10, 15) // 10*1.0 + 15*2.0 = 40
	assert.NoError(t, err)
	tokens, err = tracker.GetTokenCount(ctx, "user1")
	assert.NoError(t, err)
	assert.Equal(t, float64(40), tokens, "Token count after first update")

	tokens, err = tracker.GetTokenCount(ctx, "user2")
	assert.NoError(t, err)
	assert.Equal(t, float64(0), tokens, "Initial token count for user2 should be 0")

	tokens, err = tracker.GetTokenCount(ctx, "")
	assert.NoError(t, err)
	assert.Equal(t, float64(0), tokens, "Token count for empty user should be 0")

	tokens, _ = tracker.GetTokenCount(ctx, "nonexistent")
	assert.Equal(t, float64(0), tokens, "Token count for non-existent user should be 0")
}

func TestSlidingWindowTokenTracker_WindowBehavior(t *testing.T) {
	config := DefaultVTCConfig()
	tracker := NewInMemorySlidingWindowTokenTracker(&config, WithWindowSize(100), WithTimeUnit(Milliseconds)) // 100ms window
	ctx := context.Background()

	tokens, err := tracker.GetTokenCount(ctx, "user1")
	assert.NoError(t, err)
	assert.Equal(t, float64(0), tokens, "Initial token count should be 0")

	err = tracker.UpdateTokenCount(ctx, "user1", 10, 15) // 10*1.0 + 15*2.0 = 40
	assert.NoError(t, err)
	tokens, err = tracker.GetTokenCount(ctx, "user1")
	assert.NoError(t, err)
	assert.Equal(t, float64(40), tokens, "Token count after first update")

	// Wait to move to next time bucket
	time.Sleep(10 * time.Millisecond)
	// Add tokens in next bucket
	err = tracker.UpdateTokenCount(ctx, "user1", 5, 5) // 5*1.0 + 5*2.0 = 15
	assert.NoError(t, err)
	tokens, err = tracker.GetTokenCount(ctx, "user1")
	assert.NoError(t, err)
	assert.Equal(t, float64(55), tokens, "Sum over two buckets")

	// Wait for tokens to expire (beyond 100ms window)
	time.Sleep(110 * time.Millisecond)
	tokens, err = tracker.GetTokenCount(ctx, "user1")
	assert.NoError(t, err)
	assert.Equal(t, float64(0), tokens, "Tokens outside window should not be counted")
}

func TestSlidingWindowTokenTracker_UpdateTokenCount_WithWeights(t *testing.T) {
	config := DefaultVTCConfig()
	config.InputTokenWeight = 2.0
	config.OutputTokenWeight = 3.0
	tracker := NewInMemorySlidingWindowTokenTracker(&config, WithWindowSize(100), WithTimeUnit(Milliseconds)) // 100ms window
	ctx := context.Background()

	err := tracker.UpdateTokenCount(ctx, "user2", 2, 4) // 2*2 + 4*3 = 16
	assert.NoError(t, err)
	tokens, err := tracker.GetTokenCount(ctx, "user2")
	assert.NoError(t, err)
	assert.Equal(t, float64(16), tokens, "Weighted token count")
}

func TestSlidingWindowTokenTracker_UpdateTokenCount(t *testing.T) {
	config := DefaultVTCConfig()
	tracker := NewInMemorySlidingWindowTokenTracker(&config, WithWindowSize(100), WithTimeUnit(Milliseconds)) // 100ms window
	ctx := context.Background()

	err := tracker.UpdateTokenCount(ctx, "user1", 10, 15) // 10*1.0 + 15*2.0 = 40
	assert.NoError(t, err)
	tokens, _ := tracker.GetTokenCount(ctx, "user1")
	assert.Equal(t, float64(40), tokens, "First update")

	err = tracker.UpdateTokenCount(ctx, "user1", 5, 10) // 40 + (5*1.0 + 10*2.0) = 40 + 25 = 65
	assert.NoError(t, err)
	tokens, _ = tracker.GetTokenCount(ctx, "user1")
	assert.Equal(t, float64(65), tokens, "Second update")

	err = tracker.UpdateTokenCount(ctx, "user2", 100, 50) // 100*1.0 + 50*2.0 = 200
	assert.NoError(t, err)
	tokens, _ = tracker.GetTokenCount(ctx, "user2")
	assert.Equal(t, float64(200), tokens, "Update for user2")

	err = tracker.UpdateTokenCount(ctx, "", 5, 5)
	assert.Error(t, err, "Update with empty user should error")
}

func TestSlidingWindowTokenTracker_UpdateTokenCount_WithCustomWeights(t *testing.T) {
	config := VTCConfig{
		InputTokenWeight:  2.0,
		OutputTokenWeight: 0.5,
	}
	tracker := NewInMemorySlidingWindowTokenTracker(&config, WithWindowSize(100), WithTimeUnit(Milliseconds)) // 100ms window
	ctx := context.Background()

	err := tracker.UpdateTokenCount(ctx, "user1", 10, 20)
	assert.NoError(t, err)
	tokens, _ := tracker.GetTokenCount(ctx, "user1")
	assert.Equal(t, float64(30), tokens, "Update with custom weights")

	err = tracker.UpdateTokenCount(ctx, "user1", 5, 10)
	assert.NoError(t, err)
	tokens, _ = tracker.GetTokenCount(ctx, "user1")
	assert.Equal(t, float64(45), tokens, "Second update with custom weights")
}

func TestTokenTrackerInterface(t *testing.T) {
	config := DefaultVTCConfig()

	var tracker TokenTracker = NewInMemorySlidingWindowTokenTracker(&config)

	ctx := context.Background()
	_, err := tracker.GetTokenCount(ctx, "user")
	assert.NoError(t, err)

	err = tracker.UpdateTokenCount(ctx, "user", 10, 20)
	assert.NoError(t, err)
}

func TestTotalTokenCalculationDuringPruning(t *testing.T) {
	config := DefaultVTCConfig()
	tracker := NewInMemorySlidingWindowTokenTracker(&config, WithWindowSize(100), WithTimeUnit(Milliseconds))
	ctx := context.Background()

	err := tracker.UpdateTokenCount(ctx, "user1", 10, 0)
	assert.NoError(t, err)
	err = tracker.UpdateTokenCount(ctx, "user2", 20, 0)
	assert.NoError(t, err)

	t1, err := tracker.GetTokenCount(ctx, "user1")
	assert.NoError(t, err)
	assert.Equal(t, float64(10), t1, "user1 token count")
	t2, err := tracker.GetTokenCount(ctx, "user2")
	assert.NoError(t, err)
	assert.Equal(t, float64(20), t2, "user2 token count")

	total := t1 + t2
	assert.Equal(t, float64(30), total, "combined token count")

	time.Sleep(110 * time.Millisecond)

	t1, err = tracker.GetTokenCount(ctx, "user1")
	assert.NoError(t, err)
	assert.Equal(t, float64(0), t1, "user1 tokens expired")
	t2, err = tracker.GetTokenCount(ctx, "user2")
	assert.NoError(t, err)
	assert.Equal(t, float64(0), t2, "user2 tokens expired")
}

func TestGetMinMaxTokenCount(t *testing.T) {
	config := DefaultVTCConfig()
	tracker := NewInMemorySlidingWindowTokenTracker(&config, WithWindowSize(100), WithTimeUnit(Milliseconds))
	ctx := context.Background()

	minVal, err := tracker.GetMinTokenCount(ctx)
	assert.NoError(t, err)
	assert.Equal(t, defaultTokenTrackerMinTokens, minVal, "default min tokens")
	maxVal, err := tracker.GetMaxTokenCount(ctx)
	assert.NoError(t, err)
	assert.Equal(t, defaultTokenTrackerMaxTokens, maxVal, "default max tokens")

	err = tracker.UpdateTokenCount(ctx, "user1", 500, 0)
	assert.NoError(t, err)
	minVal, _ = tracker.GetMinTokenCount(ctx)
	maxVal, _ = tracker.GetMaxTokenCount(ctx)
	assert.Equal(t, float64(500), minVal, "min after user1 update")
	assert.Equal(t, float64(500), maxVal, "max after user1 update")

	err = tracker.UpdateTokenCount(ctx, "user2", 1000, 0)
	assert.NoError(t, err)
	minVal, _ = tracker.GetMinTokenCount(ctx)
	maxVal, _ = tracker.GetMaxTokenCount(ctx)
	assert.Equal(t, float64(500), minVal, "min after user2 update")
	assert.Equal(t, float64(1000), maxVal, "max after user2 update")
}

func TestTokenTrackerThreadSafety(t *testing.T) {
	config := DefaultVTCConfig()
	tracker := NewInMemorySlidingWindowTokenTracker(&config, WithWindowSize(100), WithTimeUnit(Milliseconds))
	ctx := context.Background()

	// Number of concurrent goroutines
	const numGoroutines = 10
	// Number of operations per goroutine
	const opsPerGoroutine = 100

	// Use a WaitGroup to coordinate goroutines
	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	// Start multiple goroutines to update and read token counts concurrently
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()

			// Each goroutine uses its own user ID
			userID := fmt.Sprintf("user-%d", id)

			for j := 0; j < opsPerGoroutine; j++ {
				// Alternate between read and write operations
				if j%2 == 0 {
					// Update token count
					err := tracker.UpdateTokenCount(ctx, userID, float64(j), float64(j))
					assert.NoError(t, err)
				} else {
					// Read token count
					_, err := tracker.GetTokenCount(ctx, userID)
					assert.NoError(t, err)
				}
			}
		}(i)
	}

	// Wait for all goroutines to complete
	wg.Wait()

	// Verify that all users have the expected token counts
	for i := 0; i < numGoroutines; i++ {
		userID := fmt.Sprintf("user-%d", i)
		tokens, err := tracker.GetTokenCount(ctx, userID)
		assert.NoError(t, err)

		// Calculate expected tokens: sum of all even j values from 0 to opsPerGoroutine-1
		// Each update adds j input tokens and j output tokens with weights from config
		expectedTokens := 0.0
		for j := 0; j < opsPerGoroutine; j += 2 {
			expectedTokens += float64(j)*config.InputTokenWeight + float64(j)*config.OutputTokenWeight
		}

		assert.Equal(t, expectedTokens, tokens, "Token count for %s should match expected value", userID)
	}

	// Also test min/max functions
	min, err := tracker.GetMinTokenCount(ctx)
	assert.NoError(t, err)
	max, err := tracker.GetMaxTokenCount(ctx)
	assert.NoError(t, err)
	assert.True(t, min <= max, "Min token count should be less than or equal to max token count")
}

func TestTokenTrackerThreadSafety_SharedUser(t *testing.T) {
	config := DefaultVTCConfig()
	tracker := NewInMemorySlidingWindowTokenTracker(&config, WithWindowSize(100), WithTimeUnit(Milliseconds))
	ctx := context.Background()

	// Number of concurrent goroutines all updating the same user
	const numGoroutines = 20
	// Number of operations per goroutine
	const opsPerGoroutine = 50
	// All goroutines update the same user
	const sharedUserID = "shared-user"

	// Use atomic counter to track the expected total
	var expectedTotal int64 = 0

	// Use a WaitGroup to coordinate goroutines
	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	// Start multiple goroutines to update the same user's tokens concurrently
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()

			for j := 0; j < opsPerGoroutine; j++ {
				// Each goroutine adds a fixed amount of tokens
				inputTokens := float64(id + 1)
				outputTokens := float64(id + 1)

				err := tracker.UpdateTokenCount(ctx, sharedUserID, inputTokens, outputTokens)
				assert.NoError(t, err)

				// Track expected total with atomic operations
				atomic.AddInt64(&expectedTotal, int64(inputTokens*config.InputTokenWeight+outputTokens*config.OutputTokenWeight))
			}
		}(i)
	}

	// Wait for all goroutines to complete
	wg.Wait()

	// Verify the final token count
	tokens, err := tracker.GetTokenCount(ctx, sharedUserID)
	assert.NoError(t, err)
	assert.Equal(t, float64(expectedTotal), tokens, "Token count for shared user should match expected value")
}

func TestTokenTrackerThreadSafety_Expiration(t *testing.T) {
	config := DefaultVTCConfig()
	// Use a very short window to test expiration
	tracker := NewInMemorySlidingWindowTokenTracker(&config, WithWindowSize(20), WithTimeUnit(Milliseconds))
	ctx := context.Background()

	// Number of concurrent goroutines
	const numGoroutines = 10
	// Number of operations per goroutine
	const opsPerGoroutine = 20

	// Use a WaitGroup to coordinate goroutines
	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	// Start multiple goroutines to update and read token counts with expiration
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()

			userID := fmt.Sprintf("exp-user-%d", id)

			for j := 0; j < opsPerGoroutine; j++ {
				// Add tokens
				err := tracker.UpdateTokenCount(ctx, userID, 1.0, 1.0)
				assert.NoError(t, err)

				// Sleep to allow some tokens to expire (stagger the sleeps)
				if j%5 == 0 {
					time.Sleep(time.Duration(5+id) * time.Millisecond)
				}

				// Read token count
				_, err = tracker.GetTokenCount(ctx, userID)
				assert.NoError(t, err)
			}
		}(i)
	}

	// Wait for all goroutines to complete
	wg.Wait()

	// Wait for all tokens to expire
	time.Sleep(30 * time.Millisecond)

	// Verify all tokens expired
	for i := 0; i < numGoroutines; i++ {
		userID := fmt.Sprintf("exp-user-%d", i)
		tokens, err := tracker.GetTokenCount(ctx, userID)
		assert.NoError(t, err)
		assert.Equal(t, 0.0, tokens, "All tokens should have expired")
	}
}

func TestTokenTrackerThreadSafety_MinMaxRecalculation(t *testing.T) {
	config := DefaultVTCConfig()
	tracker := NewInMemorySlidingWindowTokenTracker(&config, WithWindowSize(100), WithTimeUnit(Milliseconds))
	ctx := context.Background()

	// Number of concurrent goroutines
	const numGoroutines = 10

	// Use a WaitGroup to coordinate goroutines
	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	// Start multiple goroutines to update token counts with values that will trigger min/max recalculations
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()

			// Each goroutine uses a different user
			userID := fmt.Sprintf("minmax-user-%d", id)

			// Add a specific token count based on the goroutine ID
			tokenValue := float64(100 * (id + 1))
			err := tracker.UpdateTokenCount(ctx, userID, tokenValue, 0)
			assert.NoError(t, err)

			// Get min/max to trigger potential race conditions
			_, err = tracker.GetMinTokenCount(ctx)
			assert.NoError(t, err)
			_, err = tracker.GetMaxTokenCount(ctx)
			assert.NoError(t, err)

			// Sleep a bit to stagger operations
			time.Sleep(time.Duration(id) * time.Millisecond)

			// Remove the tokens to trigger min/max recalculation
			time.Sleep(110 * time.Millisecond) // Wait for tokens to expire

			// Add a different token count
			newTokenValue := float64(50 * (id + 1))
			err = tracker.UpdateTokenCount(ctx, userID, newTokenValue, 0)
			assert.NoError(t, err)
		}(i)
	}

	// Wait for all goroutines to complete
	wg.Wait()

	// Verify min and max are consistent
	min, err := tracker.GetMinTokenCount(ctx)
	assert.NoError(t, err)
	max, err := tracker.GetMaxTokenCount(ctx)
	assert.NoError(t, err)
	assert.True(t, min <= max, "Min token count should be less than or equal to max token count")

	// The expected min should be 50 (from user-0)
	expectedMin := 50.0
	// The expected max should be 500 (from user-9)
	expectedMax := 50.0 * float64(numGoroutines)

	assert.Equal(t, expectedMin, min, "Min token count should match expected value")
	assert.Equal(t, expectedMax, max, "Max token count should match expected value")
}

func TestTokenExpirationScenarios(t *testing.T) {
	tests := []struct {
		name         string
		setupFunc    func(TokenTracker, context.Context) error
		verifyFunc   func(TokenTracker, context.Context, *testing.T)
		expiryWaitMs int
	}{
		{
			name: "MultipleUsersExpiration",
			setupFunc: func(tracker TokenTracker, ctx context.Context) error {
				// Add tokens for multiple users
				err := tracker.UpdateTokenCount(ctx, "user1", 100, 0)
				if err != nil {
					return err
				}
				return tracker.UpdateTokenCount(ctx, "user2", 200, 0)
			},
			verifyFunc: func(tracker TokenTracker, ctx context.Context, t *testing.T) {
				// Verify min and max are set correctly before expiration
				min, err := tracker.GetMinTokenCount(ctx)
				assert.NoError(t, err)
				assert.Equal(t, float64(100), min, "min should be 100")
				max, err := tracker.GetMaxTokenCount(ctx)
				assert.NoError(t, err)
				assert.Equal(t, float64(200), max, "max should be 200")

				// After all tokens expire, GetTokenCount should return 0 for both users
				tokensUser1, err := tracker.GetTokenCount(ctx, "user1")
				assert.NoError(t, err)
				assert.Equal(t, float64(0), tokensUser1, "user1 tokens should be 0 after expiration")
				tokensUser2, err := tracker.GetTokenCount(ctx, "user2")
				assert.NoError(t, err)
				assert.Equal(t, float64(0), tokensUser2, "user2 tokens should be 0 after expiration")
			},
			expiryWaitMs: 110,
		},
		{
			name: "SingleUserExpiration",
			setupFunc: func(tracker TokenTracker, ctx context.Context) error {
				// Add tokens for a single user
				return tracker.UpdateTokenCount(ctx, "user1", 100, 0)
			},
			verifyFunc: func(tracker TokenTracker, ctx context.Context, t *testing.T) {
				// Verify min and max are set correctly before expiration
				min, err := tracker.GetMinTokenCount(ctx)
				assert.NoError(t, err)
				assert.Equal(t, float64(100), min, "min should be 100")
				max, err := tracker.GetMaxTokenCount(ctx)
				assert.NoError(t, err)
				assert.Equal(t, float64(100), max, "max should be 100")

				// After tokens expire, token count should be 0
				tokens, err := tracker.GetTokenCount(ctx, "user1")
				assert.NoError(t, err)
				assert.Equal(t, float64(0), tokens, "tokens should be 0 after expiration")
			},
			expiryWaitMs: 110,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			config := DefaultVTCConfig()
			tracker := NewInMemorySlidingWindowTokenTracker(&config, WithWindowSize(100), WithTimeUnit(Milliseconds))
			ctx := context.Background()

			// Run setup
			err := tc.setupFunc(tracker, ctx)
			assert.NoError(t, err)

			// Wait for tokens to expire
			time.Sleep(time.Duration(tc.expiryWaitMs) * time.Millisecond)

			// Verify the result
			tc.verifyFunc(tracker, ctx, t)
		})
	}
}

func TestTokenTrackerEdgeCases(t *testing.T) {
	tests := []struct {
		name           string
		setupFunc      func(TokenTracker, context.Context) error
		updateFunc     func(TokenTracker, context.Context) error
		expectedTokens float64
		message        string
	}{
		{
			name: "ZeroTokenUpdateIsNoOp",
			setupFunc: func(tracker TokenTracker, ctx context.Context) error {
				return tracker.UpdateTokenCount(ctx, "user1", 50, 25) // 50 + 25*2 = 100
			},
			updateFunc: func(tracker TokenTracker, ctx context.Context) error {
				return tracker.UpdateTokenCount(ctx, "user1", 0, 0)
			},
			expectedTokens: 100,
			message:        "token count should be unchanged after zero-token update",
		},
		{
			name: "SameBucketAccumulation",
			setupFunc: func(tracker TokenTracker, ctx context.Context) error {
				return tracker.UpdateTokenCount(ctx, "user1", 5, 0)
			},
			updateFunc: func(tracker TokenTracker, ctx context.Context) error {
				return tracker.UpdateTokenCount(ctx, "user1", 7, 0)
			},
			expectedTokens: 12,
			message:        "tokens should accumulate in the same time bucket",
		},
		{
			name: "NegativeTokensClampToZero",
			setupFunc: func(tracker TokenTracker, ctx context.Context) error {
				// No setup needed
				return nil
			},
			updateFunc: func(tracker TokenTracker, ctx context.Context) error {
				return tracker.UpdateTokenCount(ctx, "user1", -5, 0)
			},
			expectedTokens: 0,
			message:        "negative tokens should be clamped to zero",
		},
		{
			name: "NegativeTokenUpdateIsNoOp",
			setupFunc: func(tracker TokenTracker, ctx context.Context) error {
				return tracker.UpdateTokenCount(ctx, "user1", 50, 25) // 50 + 25*2 = 100
			},
			updateFunc: func(tracker TokenTracker, ctx context.Context) error {
				// Update with negative tokens should be a no-op
				err := tracker.UpdateTokenCount(ctx, "user1", -10, -5)
				if err != nil {
					return err
				}
				// Multiple negative updates also shouldn't change anything
				return tracker.UpdateTokenCount(ctx, "user1", -20, -15)
			},
			expectedTokens: 100,
			message:        "token count should be unchanged after negative token updates",
		},
		{
			name: "PositiveAfterNegativeTokens",
			setupFunc: func(tracker TokenTracker, ctx context.Context) error {
				// First add negative tokens (should be clamped to 0)
				err := tracker.UpdateTokenCount(ctx, "user1", -10, -5)
				if err != nil {
					return err
				}
				return nil
			},
			updateFunc: func(tracker TokenTracker, ctx context.Context) error {
				// Then add positive tokens
				return tracker.UpdateTokenCount(ctx, "user1", 30, 10) // 30 + 10*2 = 50
			},
			expectedTokens: 50,
			message:        "positive tokens should be added correctly after negative tokens",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			config := DefaultVTCConfig()
			tracker := NewInMemorySlidingWindowTokenTracker(&config, WithWindowSize(100), WithTimeUnit(Milliseconds))
			ctx := context.Background()

			// Run setup
			err := tc.setupFunc(tracker, ctx)
			assert.NoError(t, err)

			// Run the update function
			err = tc.updateFunc(tracker, ctx)
			assert.NoError(t, err)

			// Verify the result
			tokens, err := tracker.GetTokenCount(ctx, "user1")
			assert.NoError(t, err)
			assert.Equal(t, tc.expectedTokens, tokens, tc.message)
		})
	}
}

func TestSlidingWindowTokenTracker_SecondsUnitWindow(t *testing.T) {
	config := DefaultVTCConfig()
	tracker := NewInMemorySlidingWindowTokenTracker(&config, WithWindowSize(1), WithTimeUnit(Seconds)) // 1s window
	ctx := context.Background()

	err := tracker.UpdateTokenCount(ctx, "user", 1, 0)
	assert.NoError(t, err)
	toks, err := tracker.GetTokenCount(ctx, "user")
	assert.NoError(t, err)
	assert.Equal(t, float64(1), toks, "initial token count in seconds window")

	// wait beyond 1 second (account for second-level granularity)
	time.Sleep(2100 * time.Millisecond)
	toks, err = tracker.GetTokenCount(ctx, "user")
	assert.NoError(t, err)
	assert.Equal(t, float64(0), toks, "token expired after seconds window")
}

func TestTokenTrackerWindowSizeThroughConstructor(t *testing.T) {
	config := &VTCConfig{
		InputTokenWeight:  1.0,
		OutputTokenWeight: 1.0,
	}

	// Create tracker using constructor
	tracker := NewInMemorySlidingWindowTokenTracker(config)
	vtcTracker := tracker.(*InMemorySlidingWindowTokenTracker)
	windowSize := vtcTracker.windowSize
	bucketUnit := vtcTracker.bucketUnit

	t.Logf("Initial state - windowSize: %v, bucketUnit: %v", windowSize, bucketUnit)

	if windowSize == 0 {
		t.Errorf("Window size should not be 0, got %v", windowSize)
	}
}

func TestTokenTrackerMinMaxBugAfterExpiration(t *testing.T) {
	config := DefaultVTCConfig()
	tracker := NewInMemorySlidingWindowTokenTracker(&config, WithWindowSize(100), WithTimeUnit(Milliseconds))
	ctx := context.Background()

	// Initially, min/max should be defaults since no users have tokens
	minVal, err := tracker.GetMinTokenCount(ctx)
	assert.NoError(t, err)
	assert.Equal(t, defaultTokenTrackerMinTokens, minVal, "initial min should be default")
	maxVal, err := tracker.GetMaxTokenCount(ctx)
	assert.NoError(t, err)
	assert.Equal(t, defaultTokenTrackerMaxTokens, maxVal, "initial max should be default")

	// Add tokens for multiple users with different amounts
	err = tracker.UpdateTokenCount(ctx, "user1", 50, 0) // 50 tokens
	assert.NoError(t, err)
	err = tracker.UpdateTokenCount(ctx, "user2", 200, 0) // 200 tokens
	assert.NoError(t, err)
	err = tracker.UpdateTokenCount(ctx, "user3", 100, 0) // 100 tokens
	assert.NoError(t, err)

	// Verify min/max are tracking active users correctly
	minVal, err = tracker.GetMinTokenCount(ctx)
	assert.NoError(t, err)
	assert.Equal(t, float64(50), minVal, "min should be 50 (user1)")
	maxVal, err = tracker.GetMaxTokenCount(ctx)
	assert.NoError(t, err)
	assert.Equal(t, float64(200), maxVal, "max should be 200 (user2)")

	// Verify all users have correct token counts
	tokens1, err := tracker.GetTokenCount(ctx, "user1")
	assert.NoError(t, err)
	assert.Equal(t, float64(50), tokens1, "user1 should have 50 tokens")
	tokens2, err := tracker.GetTokenCount(ctx, "user2")
	assert.NoError(t, err)
	assert.Equal(t, float64(200), tokens2, "user2 should have 200 tokens")
	tokens3, err := tracker.GetTokenCount(ctx, "user3")
	assert.NoError(t, err)
	assert.Equal(t, float64(100), tokens3, "user3 should have 100 tokens")

	// Wait for ALL tokens to expire (beyond 100ms window)
	time.Sleep(110 * time.Millisecond)

	// After expiration, GetTokenCount should return 0 for all users (this works correctly)
	tokens1, err = tracker.GetTokenCount(ctx, "user1")
	assert.NoError(t, err)
	assert.Equal(t, float64(0), tokens1, "user1 tokens should be 0 after expiration")
	tokens2, err = tracker.GetTokenCount(ctx, "user2")
	assert.NoError(t, err)
	assert.Equal(t, float64(0), tokens2, "user2 tokens should be 0 after expiration")
	tokens3, err = tracker.GetTokenCount(ctx, "user3")
	assert.NoError(t, err)
	assert.Equal(t, float64(0), tokens3, "user3 tokens should be 0 after expiration")

	// 🐛 BUG EXPOSURE: After ALL users expire, min/max should return to defaults
	// but they will still return stale values from before expiration
	minVal, err = tracker.GetMinTokenCount(ctx)
	assert.NoError(t, err)
	// This assertion will FAIL, exposing the bug:
	// Expected: defaultTokenTrackerMinTokens (1000.0)
	// Actual: 50 (stale value from user1)
	assert.Equal(t, defaultTokenTrackerMinTokens, minVal,
		"BUG: min should return to default (1000) when all users expired, but returns stale value")

	maxVal, err = tracker.GetMaxTokenCount(ctx)
	assert.NoError(t, err)
	// This assertion will FAIL, exposing the bug:
	// Expected: defaultTokenTrackerMaxTokens (8000.0)
	// Actual: 200 (stale value from user2)
	assert.Equal(t, defaultTokenTrackerMaxTokens, maxVal,
		"BUG: max should return to default (8000) when all users expired, but returns stale value")

	// Additional verification: Add a new user after expiration
	// This should reset the min/max to the new user's value
	err = tracker.UpdateTokenCount(ctx, "user4", 75, 0) // 75 tokens
	assert.NoError(t, err)

	// Now min and max should both be 75 (only active user)
	minVal, err = tracker.GetMinTokenCount(ctx)
	assert.NoError(t, err)
	assert.Equal(t, float64(75), minVal, "min should be 75 after adding new user")
	maxVal, err = tracker.GetMaxTokenCount(ctx)
	assert.NoError(t, err)
	assert.Equal(t, float64(75), maxVal, "max should be 75 after adding new user")
}

func TestTokenTrackerRealWorldScenario(t *testing.T) {
	config := DefaultVTCConfig()
	// Use seconds instead of milliseconds to better simulate real world
	tracker := NewInMemorySlidingWindowTokenTracker(&config, WithWindowSize(5), WithTimeUnit(Seconds))
	ctx := context.Background()

	t.Logf("=== Initial State ===")
	minVal, _ := tracker.GetMinTokenCount(ctx)
	maxVal, _ := tracker.GetMaxTokenCount(ctx)
	t.Logf("Initial: min=%f, max=%f", minVal, maxVal)

	// Simulate a benchmark run with varying token counts like in real scenario
	t.Logf("=== Simulating Benchmark Run ===")
	err := tracker.UpdateTokenCount(ctx, "user-small-2", 41, 20) // 41 + 20*2 = 81 tokens (like real data)
	assert.NoError(t, err)
	err = tracker.UpdateTokenCount(ctx, "user-med-2", 202, 20) // 202 + 20*2 = 242 tokens
	assert.NoError(t, err)
	err = tracker.UpdateTokenCount(ctx, "user-high-1", 292, 20) // 292 + 20*2 = 332 tokens
	assert.NoError(t, err)
	err = tracker.UpdateTokenCount(ctx, "user-high-2", 346, 20) // 346 + 20*2 = 386 tokens
	assert.NoError(t, err)

	minVal, _ = tracker.GetMinTokenCount(ctx)
	maxVal, _ = tracker.GetMaxTokenCount(ctx)
	adaptiveBucketSize := math.Max(1000, (minVal+maxVal)/2)
	t.Logf("After benchmark: min=%f, max=%f, adaptiveBucketSize=%f", minVal, maxVal, adaptiveBucketSize)

	// Verify individual token counts
	tokens := make(map[string]float64)
	users := []string{"user-small-2", "user-med-2", "user-high-1", "user-high-2"}
	for _, user := range users {
		tokenCount, _ := tracker.GetTokenCount(ctx, user)
		tokens[user] = tokenCount
		t.Logf("User %s: %f tokens", user, tokenCount)
	}

	// Wait 6 seconds (beyond the 5-second window) - simulate 45 minutes later
	t.Logf("=== Waiting 6 seconds (beyond 5s window) ===")
	time.Sleep(6 * time.Second)

	// Check token counts after expiration
	t.Logf("=== After Token Expiration ===")
	allExpired := true
	for _, user := range users {
		tokenCount, _ := tracker.GetTokenCount(ctx, user)
		t.Logf("User %s: %f tokens (after expiration)", user, tokenCount)
		if tokenCount != 0 {
			allExpired = false
		}
	}
	assert.True(t, allExpired, "All user tokens should be 0 after window expiration")

	// Check min/max after expiration - this is the critical test
	minVal, _ = tracker.GetMinTokenCount(ctx)
	maxVal, _ = tracker.GetMaxTokenCount(ctx)
	adaptiveBucketSize = math.Max(1000, (minVal+maxVal)/2)
	t.Logf("After expiration: min=%f, max=%f, adaptiveBucketSize=%f", minVal, maxVal, adaptiveBucketSize)

	// This is what causes the bug in real world:
	// If min/max don't reset, bucket size stays high
	if minVal != 1000.0 || maxVal != 8000.0 {
		t.Errorf("BUG FOUND: min/max should reset to defaults when all users expire")
		t.Errorf("Expected: min=1000, max=8000")
		t.Errorf("Actual: min=%f, max=%f", minVal, maxVal)
		t.Errorf("This causes adaptiveBucketSize=%f instead of expected ~4500", adaptiveBucketSize)
	}

	// Additional test: Simulate the exact VTC algorithm calculation
	expectedBucketSizeIfFixed := math.Max(1000, (1000+8000)/2) // = 4500
	t.Logf("Expected bucket size if fixed: %f", expectedBucketSizeIfFixed)
	t.Logf("Actual bucket size with current logic: %f", adaptiveBucketSize)
}

func TestTokenTrackerLazyPruningBug(t *testing.T) {
	config := DefaultVTCConfig()
	tracker := NewInMemorySlidingWindowTokenTracker(&config, WithWindowSize(100), WithTimeUnit(Milliseconds))
	ctx := context.Background()

	t.Logf("=== Initial State ===")
	minVal, _ := tracker.GetMinTokenCount(ctx)
	maxVal, _ := tracker.GetMaxTokenCount(ctx)
	t.Logf("Initial: min=%f, max=%f", minVal, maxVal)

	// Add tokens for users
	err := tracker.UpdateTokenCount(ctx, "user1", 50, 0) // 50 tokens
	assert.NoError(t, err)
	err = tracker.UpdateTokenCount(ctx, "user2", 200, 0) // 200 tokens
	assert.NoError(t, err)

	minVal, _ = tracker.GetMinTokenCount(ctx)
	maxVal, _ = tracker.GetMaxTokenCount(ctx)
	t.Logf("After adding users: min=%f, max=%f", minVal, maxVal)

	// Wait for tokens to expire
	time.Sleep(110 * time.Millisecond)

	// 🐛 KEY DIFFERENCE: Don't call GetTokenCount() for any users
	// In real world, if no users are making requests, GetTokenCount() is never called
	// This means pruning never happens, so min/max tracking doesn't get updated

	// Now call GetMinTokenCount and GetMaxTokenCount directly (like VTC algorithm does)
	minVal, _ = tracker.GetMinTokenCount(ctx)
	maxVal, _ = tracker.GetMaxTokenCount(ctx)
	t.Logf("After expiration WITHOUT calling GetTokenCount: min=%f, max=%f", minVal, maxVal)

	// 🐛 BUG EXPOSURE: These should be defaults (1000, 8000) but will be stale (50, 200)
	// because no GetTokenCount() calls triggered pruning
	if minVal == 50.0 && maxVal == 200.0 {
		t.Errorf("BUG CONFIRMED: min/max are stale because pruning is lazy!")
		t.Errorf("GetMinTokenCount/GetMaxTokenCount don't trigger pruning")
		t.Errorf("They only use cached minTrackedToken/maxTrackedToken values")
		t.Errorf("Pruning only happens in GetTokenCount() and UpdateTokenCount()")
	}

	// Now call GetTokenCount() for users to trigger pruning
	t.Logf("=== Triggering pruning by calling GetTokenCount ===")
	tokens1, _ := tracker.GetTokenCount(ctx, "user1")
	tokens2, _ := tracker.GetTokenCount(ctx, "user2")
	t.Logf("User1: %f, User2: %f (after pruning)", tokens1, tokens2)

	// NOW check min/max again - they should be fixed
	minVal, _ = tracker.GetMinTokenCount(ctx)
	maxVal, _ = tracker.GetMaxTokenCount(ctx)
	t.Logf("After pruning triggered: min=%f, max=%f", minVal, maxVal)

	// Verify they're now correct
	assert.Equal(t, defaultTokenTrackerMinTokens, minVal, "min should be default after pruning")
	assert.Equal(t, defaultTokenTrackerMaxTokens, maxVal, "max should be default after pruning")
}
