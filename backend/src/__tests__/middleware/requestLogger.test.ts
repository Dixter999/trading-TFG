import { Request, Response, NextFunction } from 'express';
import { requestLogger, logger } from '../../middleware/requestLogger';

// Extend Request type to include id
interface RequestWithId extends Request {
  id?: string;
}

// Mock winston logger
jest.mock('winston', () => ({
  createLogger: jest.fn(() => ({
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
  })),
  format: {
    combine: jest.fn(),
    timestamp: jest.fn(),
    json: jest.fn(),
    printf: jest.fn(),
  },
  transports: {
    Console: jest.fn(),
    File: jest.fn(),
  },
}));

describe('Request Logger Middleware', () => {
  let mockRequest: Partial<RequestWithId>;
  let mockResponse: Partial<Response>;
  let nextFunction: NextFunction;
  let loggerInfoSpy: jest.SpyInstance;

  beforeEach(() => {
    mockRequest = {
      method: 'GET',
      path: '/api/markets/EURUSD/H1',
      query: { start: '2024-01-01T00:00:00Z', limit: '1000' },
      headers: {
        'user-agent': 'test-agent',
        'content-type': 'application/json',
      },
      id: 'req_test123',
    };
    mockResponse = {
      statusCode: 200,
      getHeader: jest.fn(),
      on: jest.fn(),
    };
    nextFunction = jest.fn();
    loggerInfoSpy = jest.spyOn(logger, 'info');
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Incoming request logging', () => {
    it('should log incoming request with method and path', () => {
      requestLogger(mockRequest as RequestWithId, mockResponse as Response, nextFunction);

      expect(loggerInfoSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'request',
          method: 'GET',
          path: '/api/markets/EURUSD/H1',
        })
      );
    });

    it('should log query parameters', () => {
      requestLogger(mockRequest as RequestWithId, mockResponse as Response, nextFunction);

      expect(loggerInfoSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          query: { start: '2024-01-01T00:00:00Z', limit: '1000' },
        })
      );
    });

    it('should log request headers', () => {
      requestLogger(mockRequest as RequestWithId, mockResponse as Response, nextFunction);

      expect(loggerInfoSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          headers: expect.objectContaining({
            'user-agent': 'test-agent',
            'content-type': 'application/json',
          }),
        })
      );
    });

    it('should log requestId', () => {
      requestLogger(mockRequest as RequestWithId, mockResponse as Response, nextFunction);

      expect(loggerInfoSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          requestId: 'req_test123',
        })
      );
    });

    it('should log timestamp for incoming request', () => {
      requestLogger(mockRequest as RequestWithId, mockResponse as Response, nextFunction);

      expect(loggerInfoSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          timestamp: expect.any(String),
        })
      );
    });
  });

  describe('Response logging', () => {
    it('should log response status code', () => {
      requestLogger(mockRequest as RequestWithId, mockResponse as Response, nextFunction);

      // Get the finish event handler
      const finishHandler = (mockResponse.on as jest.Mock).mock.calls.find(
        (call) => call[0] === 'finish'
      )?.[1];

      expect(finishHandler).toBeDefined();

      // Trigger the finish event
      finishHandler();

      // Check that logger.info was called with status code
      const responseLogs = loggerInfoSpy.mock.calls.filter((call) => call[0].type === 'response');
      expect(responseLogs.length).toBeGreaterThan(0);
      expect(responseLogs[0][0]).toMatchObject({
        statusCode: 200,
      });
    });

    it('should log response duration in milliseconds', () => {
      requestLogger(mockRequest as RequestWithId, mockResponse as Response, nextFunction);

      const finishHandler = (mockResponse.on as jest.Mock).mock.calls.find(
        (call) => call[0] === 'finish'
      )?.[1];

      finishHandler();

      const responseLogs = loggerInfoSpy.mock.calls.filter((call) => call[0].type === 'response');
      expect(responseLogs[0][0]).toMatchObject({
        duration: expect.any(Number),
      });
      expect(responseLogs[0][0].duration).toBeGreaterThanOrEqual(0);
    });

    it('should log response size in bytes from content-length header', () => {
      (mockResponse.getHeader as jest.Mock).mockReturnValue('1024');

      requestLogger(mockRequest as RequestWithId, mockResponse as Response, nextFunction);

      const finishHandler = (mockResponse.on as jest.Mock).mock.calls.find(
        (call) => call[0] === 'finish'
      )?.[1];

      finishHandler();

      const responseLogs = loggerInfoSpy.mock.calls.filter((call) => call[0].type === 'response');
      expect(responseLogs[0][0]).toMatchObject({
        bytes: 1024,
      });
    });

    it('should handle missing content-length header', () => {
      (mockResponse.getHeader as jest.Mock).mockReturnValue(undefined);

      requestLogger(mockRequest as RequestWithId, mockResponse as Response, nextFunction);

      const finishHandler = (mockResponse.on as jest.Mock).mock.calls.find(
        (call) => call[0] === 'finish'
      )?.[1];

      finishHandler();

      const responseLogs = loggerInfoSpy.mock.calls.filter((call) => call[0].type === 'response');
      expect(responseLogs[0][0]).toMatchObject({
        bytes: 0,
      });
    });

    it('should include requestId in response log', () => {
      requestLogger(mockRequest as RequestWithId, mockResponse as Response, nextFunction);

      const finishHandler = (mockResponse.on as jest.Mock).mock.calls.find(
        (call) => call[0] === 'finish'
      )?.[1];

      finishHandler();

      const responseLogs = loggerInfoSpy.mock.calls.filter((call) => call[0].type === 'response');
      expect(responseLogs[0][0]).toMatchObject({
        requestId: 'req_test123',
      });
    });
  });

  describe('Structured logging', () => {
    it('should use JSON format for logs', () => {
      requestLogger(mockRequest as RequestWithId, mockResponse as Response, nextFunction);

      const loggedObject = loggerInfoSpy.mock.calls[0][0];
      expect(typeof loggedObject).toBe('object');
      expect(loggedObject).not.toBeNull();
    });

    it('should include type field to distinguish log types', () => {
      requestLogger(mockRequest as RequestWithId, mockResponse as Response, nextFunction);

      expect(loggerInfoSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'request',
        })
      );
    });
  });

  describe('Middleware behavior', () => {
    it('should call next() to continue middleware chain', () => {
      requestLogger(mockRequest as RequestWithId, mockResponse as Response, nextFunction);

      expect(nextFunction).toHaveBeenCalled();
    });

    it('should not block request processing on logging errors', () => {
      loggerInfoSpy.mockImplementation(() => {
        throw new Error('Logging failed');
      });

      expect(() => {
        requestLogger(mockRequest as Request, mockResponse as Response, nextFunction);
      }).not.toThrow();

      expect(nextFunction).toHaveBeenCalled();
    });
  });

  describe('Request timing metrics', () => {
    it('should track request start time', () => {
      const startTime = Date.now();

      requestLogger(mockRequest as RequestWithId, mockResponse as Response, nextFunction);

      const finishHandler = (mockResponse.on as jest.Mock).mock.calls.find(
        (call) => call[0] === 'finish'
      )?.[1];

      // Simulate some delay
      const endTime = startTime + 100;
      jest.spyOn(Date, 'now').mockReturnValue(endTime);

      finishHandler();

      const responseLogs = loggerInfoSpy.mock.calls.filter((call) => call[0].type === 'response');
      expect(responseLogs[0][0].duration).toBeGreaterThanOrEqual(0);
    });
  });
});
